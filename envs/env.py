import os

import torch
import pickle
from os.path import join as pjoin
from typing import Iterable, Optional,Union
from tensordict.tensordict import TensorDict
from torchrl.data import BoundedTensorSpec,CompositeSpec,UnboundedContinuousTensorSpec,UnboundedDiscreteTensorSpec
from torchrl.envs import EnvBase
from torch.utils.data import DataLoader

from utils.functions import gather_by_index, get_distance, load_npz_to_tensordict
from envs.generator import MTVRPGenerator

def get_dataloader(dataset, batch_size, ddp=False, num_workers=0):
    # dataset: dataset /  dict {str:dataset}
    # batch_size: int / list(int)
    # return dataloader / list[dataloader]
    def get_single_dataloader(dataset_, batch_size_, ddp_=False, num_workers_=0):
        def return_x(x):
            return x
        sampler_ = torch.utils.data.distributed.DistributedSampler(dataset_, shuffle=False) if ddp_ else None
        return DataLoader(
            dataset_,
            batch_size=batch_size_,
            sampler = sampler_,
            shuffle=False,
            num_workers=num_workers_,
            collate_fn=return_x,
        )
    if isinstance(dataset, dict):
        # dataloader_num = len(list(dataset.keys()))
        # if isinstance(batch_size, int): batch_size = [batch_size] * dataloader_num
        # assert len(batch_size) == dataloader_num, f"Batch size: {len(batch_size)} and number of datasets {dataloader_num}"
        # size_bs = {50:500, 100:500, 200:150, 300:80}
        size_bs = {50: 500, 100: 1000, 200: 125, 300: 100}
        # size_bs = {50: 500, 100: 250, 200: 85, 300: 100}
        batch_size = [size_bs[int(x.split('_')[0])] for x in list(dataset.keys())]
        return {
            name: get_single_dataloader(dset, bsize, ddp, num_workers)
            for (name,dset), bsize in zip(dataset.items(), batch_size)
        }
    else:
        assert isinstance(batch_size, int), f"Found {batch_size}"
        return get_single_dataloader(dataset, batch_size, ddp, num_workers)

class MTVRPEnv(EnvBase):
    r"""MTVRPEnv is a Multi-Task VRP environment which can take any combination of the following constraints:
    Features:
    - *Capacity (C)*
        - Each vehicle has a maximum capacity :math:`Q`, restricting the total load that can be in the vehicle at any point of the route.
        - The route must be planned such that the sum of demands and pickups for all customers visited does not exceed this capacity.
    - *Time Windows (TW)*
        - Every node :math:`i` has an associated time window :math:`[e_i, l_i]` during which service must commence.
        - Additionally, each node has a service time :math:`s_i`. Vehicles must reach node :math:`i` within its time window; early arrivals must wait at the node location until time :math:`e_i`.
    - *Open Routes (O)*
        - Vehicles are not required to return to the depot after serving all customers.
        - Note that this does not need to be counted as a constraint since it can be modelled by setting zero costs on arcs returning to the depot :math:`c_{i0} = 0` from any customer :math:`i \in C`, and not counting the return arc as part of the route.
    - *Backhauls (B)*
        - Backhauls generalize demand to also account for return shipments. Customers are either linehaul or backhaul customers.
        - Linehaul customers require delivery of a demand :math:`q_i > 0` that needs to be transported from the depot to the customer, whereas backhaul customers need a pickup of an amount :math:`p_i > 0` that is transported from the client back to the depot.
        - It is possible for vehicles to serve a combination of linehaul and backhaul customers in a single route, but then any linehaul customers must precede the backhaul customers in the route.
    - *Duration Limits (L)*
        - Imposes a limit on the total travel duration (or length) of each route, ensuring a balanced workload across vehicles.
    Args:
        generator: Generator for the environment, see :class:`MTVRPGenerator`.
        generator_params: Parameters for the generator.
    """
    name = "mtvrp"
    def __init__(
            self,
            generator_params: dict = {},
            data_dir: str = "data/",
            test_size: list = None, test_problem: list = None, test_distribution: list = None,
            check_solution: bool = False, seed: int = None,
            device: str = "cpu", opt_dir:str = 'hgs', **kwargs
    ):
        super().__init__(device=device, )
        self.data_dir = data_dir
        self.opt_dir = opt_dir
        self.val_file = self.val_dataloader_names = None # no val
        # warn! will not load "test_size", "test_problem", "test_distribution", "test_file", "test_dataloader_names" from resume
        self.test_size = test_size
        self.test_problem = test_problem
        self.test_distribution = test_distribution
        self.test_file = []
        self.test_dataloader_names = []
        # self.test_file = test_file
        # self.test_dataloader_names = test_dataloader_names
        self.check_solution = check_solution
        if seed is None: seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.generator = MTVRPGenerator(**generator_params)
        self._make_spec()

    def step(self, td: TensorDict) -> TensorDict:
        td = self._step(td)
        return {"next": td}

    def _step(self, td: TensorDict) -> TensorDict:
        # Get locations and distance
        prev_node, curr_node = td["current_node"], td["action"] #prev_node batch_size*repeat, curr_node
        prev_loc = gather_by_index(td["locs"], prev_node) # batch_size*repeat
        curr_loc = gather_by_index(td["locs"], curr_node)
        distance = get_distance(prev_loc, curr_loc)[..., None]

        # Update current time
        service_time = gather_by_index(src=td["service_time"], idx=curr_node, dim=1, squeeze=False)
        start_times = gather_by_index(src=td["time_windows"], idx=curr_node, dim=1, squeeze=False)[..., 0]
        # we cannot start before we arrive and we should start at least at start times
        curr_time = (curr_node[:, None] != 0) * (torch.max(td["current_time"] + distance / td["speed"], start_times)+ service_time)
        # Update current route length (reset at depot)
        curr_route_length = (curr_node[:, None] != 0) * (td["current_route_length"] + distance)
        # Linehaul (delivery) demands
        selected_demand_linehaul = gather_by_index(td["demand_linehaul"], curr_node, dim=1, squeeze=False)
        selected_demand_backhaul = gather_by_index(td["demand_backhaul"], curr_node, dim=1, squeeze=False)
        # Backhaul (pickup) demands
        # vehicles are empty once we get to the backhauls
        used_capacity_linehaul = (curr_node[:, None] != 0) * (td["used_capacity_linehaul"] + selected_demand_linehaul)
        used_capacity_backhaul = (curr_node[:, None] != 0) * (td["used_capacity_backhaul"] + selected_demand_backhaul)
        # Done when all customers are visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done).float()  # we use the `get_reward` method to compute the reward

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """Reset function to call at the beginning of each episode"""
        if batch_size is None:
            batch_size = td.batch_size
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size).to('cuda')
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.to(td.device)
        return super().reset(td, batch_size=batch_size)

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None,) -> TensorDict:

        device = td.device
        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand_backhaul": td["demand_backhaul"],
                "demand_linehaul": td["demand_linehaul"],
                "distance_limit": td["distance_limit"],
                "service_time": td["service_time"],
                "open_route": td["open_route"],
                "time_windows": td["time_windows"],
                "vehicle_capacity": td["vehicle_capacity"],
                "capacity_original": td["capacity_original"],
                "speed": td["speed"],
                "current_node": torch.zeros((*batch_size,), dtype=torch.long, device=device),
                "current_route_length": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),  # for distance limits
                "current_time": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),  # for time windows
                "used_capacity_backhaul": torch.zeros((*batch_size, 1), device=device),  # for capacity constraints in backhaul
                "used_capacity_linehaul": torch.zeros((*batch_size, 1), device=device),  # for capacity constraints in linehaul
                "visited": torch.zeros((*batch_size, td["locs"].shape[-2]), dtype=torch.bool, device=device,),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def dataset(self, data_size=None, phase="train"):
        if phase == "train":
            td = self.generator(data_size)
            return td
        assert phase == 'test'
        if len(self.test_file)==0:
            all_test_file = os.listdir(self.data_dir)
            for file_i in all_test_file:
                spilt_file = file_i.split(".")[0].split("_")
                if spilt_file[-1] != 'hgs' and spilt_file[-1] != 'uniform' :
                    continue
                s_i, p_i, d_i = int(spilt_file[0]), spilt_file[1], spilt_file[2]
                if s_i in self.test_size and p_i in self.test_problem and d_i in self.test_distribution:
                    self.test_file.append(pjoin(self.data_dir, file_i))
                    if spilt_file[-1] != 'hgs':#d_i == 'uniform': #
                        self.test_dataloader_names.append(file_i.split(".")[0])
                    else: # xxxx_hgs.npz
                        self.test_dataloader_names.append(file_i.split(".")[0][:-4])
        dataset = {}
        for name, _f in zip(self.test_dataloader_names, self.test_file):
            td = load_npz_to_tensordict(_f).to('cpu') # tensordict # 默认应该在cpu上
            if data_size is not None and td.batch_size[0] > data_size:
                td = td[:data_size]
            #
            tmp_size = int(name.split('_')[0])
            tmp_p = name.split('_')[1]
            keep_mask = torch.zeros((td.shape[0],5), dtype=torch.bool)
            for id_, p_tag in enumerate(['c', 'o', 'tw', 'l', 'b']):
                keep_mask[:, id_] = True if p_tag in tmp_p else False
            # c will not be contained in vrptw, vrpb, vrpl!
            keep_mask[:, 0:1] = ~ keep_mask[:, 1:2] # not c is o # TODO:
            td['p_s_tag'] = torch.cat((
                keep_mask.float(),
                torch.full_like(td['open_route'], tmp_size/2000, dtype=torch.float32,device=keep_mask.device)
            ),dim=-1)
            dataset[name] = td
        return dataset # dict{str: td}

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        curr_node = td["current_node"]  # note that this was just updated!
        locs = td["locs"]
        d_ij = get_distance(
            gather_by_index(locs, curr_node)[..., None, :], locs
        )  # i (current) -> j (next)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)

        # Time constraint (TW):
        early_tw, late_tw = (
            td["time_windows"][..., 0],
            td["time_windows"][..., 1],
        )
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        # can reach in time -> only need to *start* in time
        can_reach_customer = arrival_time < late_tw
        # we must ensure that we can return to depot in time *if* route is closed
        # i.e. start time + service time + time back to depot < late_tw
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["service_time"] + (d_j0 / td["speed"])
        ) * ~td["open_route"] < late_tw[..., 0:1]

        # Distance limit (L): do not add distance to depot if open route (O)
        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"]
        )

        # Linehaul demand / delivery (C) and backhaul demand / pickup (B)
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )
        # All linehauls are visited before backhauls
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"],
                idx=curr_node,
                dim=1,
                squeeze=False,
            )
            > 0
        )

        meets_demand_constraint = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Condense constraints
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Mask depot: don't visit depot if coming from there and there are still customer nodes I can visit
        can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Function to compute the reward. Can be called by the agent to compute the reward of the current state
        This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
        """
        if self.check_solution:
            self.check_solution_validity(td, actions)
        # get_reward
        # Append depot to actions and get sequence of locations
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)  # [b, seq_len]
        loc_from = gather_by_index(td["locs"], go_from)
        loc_to = gather_by_index(td["locs"], go_to)

        # Get tour length. If route is open and goes to depot, don't count the distance
        distances = get_distance(loc_from, loc_to)  # [b, seq_len]
        tour_length = (distances * ~((go_to == 0) & td["open_route"])).sum(-1)  # [b]
        return -tour_length  # reward is negative cost

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        batch_size, n_loc = td["demand_linehaul"].size()
        locs = td["locs"]
        n_loc -= 1  # exclude depot
        sorted_pi = actions.data.sort(1)[0]

        # all customer nodes visited exactly once
        assert (
            torch.arange(1, n_loc + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, n_loc)
            == sorted_pi[:, -n_loc:]
        ).all() and (sorted_pi[:, :-n_loc] == 0).all(), "Invalid tour"

        # Distance limits (L)
        assert (td["distance_limit"] >= 0).all(), "Distance limits must be non-negative."

        # Time windows (TW)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(td["service_time"] >= 0.0), "Service time must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"
        assert torch.all(
            td["time_windows"][..., :, 0] + d_j0 + td["service_time"]
            <= td["time_windows"][..., 0, 1, None]
        ), "vehicle cannot perform service and get back to depot in time."
        # check individual time windows
        curr_time = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros(batch_size, dtype=torch.int64, device=td.device)
        curr_length = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            curr_loc = gather_by_index(td["locs"], curr_node)
            next_loc = gather_by_index(td["locs"], next_node)
            dist = get_distance(curr_loc, next_loc)

            # distance limit (L)
            curr_length = curr_length + dist * ~(
                td["open_route"].squeeze(-1) & (next_node == 0)
            )  # do not count back to depot for open route
            assert torch.all(
                curr_length <= td["distance_limit"].squeeze(-1)
            ), "Route exceeds distance limit"
            curr_length[next_node == 0] = 0.0  # reset length for depot

            curr_time = torch.max(
                curr_time + dist, gather_by_index(td["time_windows"], next_node)[..., 0]
            )
            assert torch.all(
                curr_time <= gather_by_index(td["time_windows"], next_node)[..., 1]
            ), "vehicle cannot start service before deadline"
            curr_time = curr_time + gather_by_index(td["service_time"], next_node)
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0  # reset time for depot

        # Demand constraints (C) and (B)
        # linehauls are the same as backhauls but with a different feature
        def _check_c1(feature="demand_linehaul"):
            demand = td[feature].gather(dim=1, index=actions)
            used_cap = torch.zeros_like(td[feature][:, 0])
            for ii in range(actions.size(1)):
                # reset at depot
                used_cap = used_cap * (actions[:, ii] != 0)
                used_cap += demand[:, ii]
                assert (
                    used_cap <= td["vehicle_capacity"]
                ).all(), "Used more than capacity for {}: {}".format(feature, used_cap)

        _check_c1("demand_linehaul")
        _check_c1("demand_backhaul")

    @staticmethod
    def select_start_nodes(td):
        """Select available start nodes for the environment (e.g. for POMO-based training)"""
        num_starts = td["action_mask"].shape[-1] - 1 # action_mask [bs, 1+n]
        num_loc = td["locs"].shape[-2] - 1 # num_loc [bs, 1+n, 2]
        selected = (torch.arange(num_starts, device=td.device).repeat_interleave(td.batch_size[0]) % num_loc + 1) #num_starts*bs
        # selected = selected.view(num_starts,-1).transpose(0, 1).contiguous().view(-1) # selected[0]  = arange(num_starts)
        return num_starts, selected

    def _make_spec(self):
        """Make the observation and action specs from the parameters."""
        use_low = False
        if use_low:
            self.observation_spec = CompositeSpec(
                locs=BoundedTensorSpec(
                    low=self.generator.min_loc,
                    high=self.generator.max_loc,
                    shape=(self.generator.num_loc + 1, 2),
                    dtype=torch.float32,
                    device=self.device,
                ),
                current_node=UnboundedDiscreteTensorSpec(
                    shape=(1),
                    dtype=torch.int64,
                    device=self.device,
                ),
                demand_linehaul=BoundedTensorSpec(
                    low=-self.generator.capacity,
                    high=self.generator.max_demand,
                    shape=(self.generator.num_loc, 1),  # demand is only for customers
                    dtype=torch.float32,
                    device=self.device,
                ),
                demand_backhaul=BoundedTensorSpec(
                    low=-self.generator.capacity,
                    high=self.generator.max_demand,
                    shape=(self.generator.num_loc, 1),  # demand is only for customers
                    dtype=torch.float32,
                    device=self.device,
                ),
                action_mask=UnboundedDiscreteTensorSpec(
                    shape=(self.generator.num_loc + 1, 1),
                    dtype=torch.bool,
                    device=self.device,
                ),
                shape=(),
            )
            self.action_spec = BoundedTensorSpec(
                low=0,
                high=self.generator.num_loc + 1,
                shape=(1,),
                dtype=torch.int64,
                device=self.device,
            )
            self.reward_spec = UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.float32, device=self.device
            )
            self.done_spec = UnboundedDiscreteTensorSpec(
                shape=(1,), dtype=torch.bool, device=self.device
            )
        else:
            self.observation_spec = CompositeSpec(
                locs=BoundedTensorSpec(
                    minimum=self.generator.min_loc,
                    maximum=self.generator.max_loc,
                    shape=(self.generator.num_loc + 1, 2),
                    dtype=torch.float32,
                    device=self.device,
                ),
                current_node=UnboundedDiscreteTensorSpec(
                    shape=(1),
                    dtype=torch.int64,
                    device=self.device,
                ),
                demand_linehaul=BoundedTensorSpec(
                    minimum=-self.generator.capacity,
                    maximum=self.generator.max_demand,
                    shape=(self.generator.num_loc, 1),  # demand is only for customers
                    dtype=torch.float32,
                    device=self.device,
                ),
                demand_backhaul=BoundedTensorSpec(
                    minimum=-self.generator.capacity,
                    maximum=self.generator.max_demand,
                    shape=(self.generator.num_loc, 1),  # demand is only for customers
                    dtype=torch.float32,
                    device=self.device,
                ),
                action_mask=UnboundedDiscreteTensorSpec(
                    shape=(self.generator.num_loc + 1, 1),
                    dtype=torch.bool,
                    device=self.device,
                ),
                shape=(),
            )
            self.action_spec = BoundedTensorSpec(
                minimum=0,
                maximum=self.generator.num_loc + 1,
                shape=(1,),
                dtype=torch.int64,
                device=self.device,
            )
            self.reward_spec = UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.float32, device=self.device
            )
            self.done_spec = UnboundedDiscreteTensorSpec(
                shape=(1,), dtype=torch.bool, device=self.device
            )

    @staticmethod
    def check_variants(td):
        """Check if the problem has the variants"""
        has_open = td["open_route"].squeeze(-1)
        has_tw = (td["time_windows"][:, :, 1] != float("inf")).any(-1)
        has_limit = (td["distance_limit"] != float("inf")).squeeze(-1)
        has_backhaul = (td["demand_backhaul"] != 0).any(-1)
        return has_open, has_tw, has_limit, has_backhaul

    @staticmethod
    def get_variant_names(td):
        (
            has_open,
            has_time_window,
            has_duration_limit,
            has_backhaul,
        ) = MTVRPEnv.check_variants(td)
        instance_names = []
        for o, b, l_, tw in zip(
            has_open, has_backhaul, has_duration_limit, has_time_window
        ):
            if not o and not b and not l_ and not tw:
                instance_name = "CVRP"
            else:
                instance_name = "VRP"
                if o:
                    instance_name = "O" + instance_name
                if b:
                    instance_name += "B"
                if l_:
                    instance_name += "L"
                if tw:
                    instance_name += "TW"
            instance_names.append(instance_name)
        return instance_names

    def print_presets(self):
        self.generator.print_presets()

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng

    def to(self, device):
        """Override `to` device method for safety against `None` device (may be found in `TensorDict`))"""
        if device is None:
            return self
        else:
            return super().to(device)

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state, set_seed:bool=True):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        # warn! will not load "test_size", "test_problem", "test_distribution", "test_file", "test_dataloader_names" from resume
        attributes_to_check = ["test_size", "test_problem", "test_distribution", "test_file", "test_dataloader_names"]
        for attr in attributes_to_check:
            if attr in state:
                # current_value = getattr(self, attr)  # Get the current value of the attribute
                # state_value = state[attr]  # Get the value from the loaded state
                # if current_value != state_value:
                #     print('xxxxxxxx')
                del state[attr]
        if not set_seed: # for ddp, use init method generate seed as state["rng"] is a tensor
            del state["rng"]
        self.__dict__.update(state)
        if set_seed:
            self.rng = torch.manual_seed(0) # = torch.default_generator
            self.rng.set_state(state["rng"].cpu()) # = set  torch.default_generator