import abc
import torch
import logging
from typing import Callable, Tuple, Union
from tensordict.tensordict import TensorDict
from torch.distributions import Exponential, Normal, Poisson, Uniform

from envs.distribution_utils import Cluster, Mixed, Gaussian_Mixture, Mix_Distribution, Mix_Multi_Distributions
from utils.functions import get_distance,save_tensordict_to_npz, get_distance

log = logging.getLogger()

def get_vehicle_capacity(num_loc: int) -> int:
    """Capacity should be 30 + num_loc/5 if num_loc > 20 as described in Liu et al. 2024 (POMO-MTL).
    For every N over 1000, we add 1 of capacity every 33.3 nodes to align with Ye et al. 2024 (GLOP),
    i.e. 260 at 2K nodes, 350 at 5K nodes and 500 at 10K nodes.
    Note that this serves as a demand scaler.
    """
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap

def get_sampler(
        distribution: Union[int, float, str, type, Callable],
        val_name: str = 'loc', low: float = 0, high: float = 1.0, **kwargs,):
    """Get the sampler for the variable with the given distribution.
    If kwargs are passed, they will be parsed e.g. with `val_name` + `_dist_arg` (e.g. `loc_std` for Normal distribution).

    Args:
        val_name: Name of the variable
        distribution: int/float value (as constant distribution), or string with the distribution name (supporting
            uniform, normal, exponential, and poisson) or PyTorch Distribution type or a callable function that
            returns a PyTorch Distribution
        low: Minimum value for the variable, used for Uniform distribution
        high: Maximum value for the variable, used for Uniform distribution
        kwargs: Additional arguments for the distribution

    Example:
    >>> sampler_uniform = get_sampler("loc", "uniform", 0, 1)
    >>> sampler_normal = get_sampler("loc", "normal", loc_mean=0.5, loc_std=.2)
    """
    if isinstance(distribution, (int, float)):
        return Uniform(low=distribution, high=distribution)
    elif distribution == Uniform or distribution == "uniform":
        return Uniform(low=low, high=high)
    elif distribution == Normal or distribution == "normal" or distribution == "gaussian":
        assert (kwargs.get(val_name + "_mean", None) is not None), "mean is required for Normal distribution"
        assert (kwargs.get(val_name + "_std", None) is not None), "std is required for Normal distribution"
        return Normal(loc=kwargs[val_name + "_loc"], scale=kwargs[val_name + "_scale"])
    elif distribution == Exponential or distribution == "exponential":
        assert (kwargs.get(val_name + "_rate", None) is not None), "rate is required for Exponential/Poisson distribution"
        return Exponential(rate=kwargs[val_name + "_rate"])
    elif distribution == Poisson or distribution == "poisson":
        assert (kwargs.get(val_name + "_rate", None) is not None), "rate is required for Exponential/Poisson distribution"
        return Poisson(rate=kwargs[val_name + "_rate"])
    elif distribution == "center":
        return Uniform(low=(high - low) / 2, high=(high - low) / 2)
    elif distribution == "corner":
        return Uniform(low=low, high=low)  # todo: should be also `low, high` and any other corner
    elif isinstance(distribution, Callable):
        return distribution(**kwargs)
    elif distribution == "gaussian_mixture":
        return Gaussian_Mixture(num_modes=kwargs['num_modes'], cdist=kwargs['cdist'])
    elif distribution == "cluster":
        return Cluster(kwargs['n_cluster'])
    elif distribution == "mixed":
        return Mixed(kwargs['n_cluster_mix'])
    elif distribution == "mix_distribution":
        return Mix_Distribution(kwargs['n_cluster'], kwargs['n_cluster_mix'])
    elif distribution == "mix_multi_distributions":
        return Mix_Multi_Distributions()
    else:
        raise ValueError(f"Invalid distribution type of {distribution}")

VARIANT_GENERATION_PRESETS = {
    "all":             {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
    "single_feat":     {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
    "single_feat_otw": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5, "OTW": 0.5},  # same training as Zhou et al. 2024
    "cvrp":            {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "ovrp":            {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "vrpb":            {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "vrpl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "vrptw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "ovrptw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "ovrpb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "ovrpl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "vrpbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "vrpbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "vrpltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "ovrpbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "ovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "ovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "vrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    "ovrpbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
}


class MTVRPGenerator:
    """MTVRP Generator.
    Class to generate instances of the MTVRP problem.
    If a variant is declared and Subsample is True, the generator will sample the problem based on the variant probabilities.
    By default, we use Mixed-Batch Training as in Berto et al. 2024 (RouteFinder), i.e. one batch can contain multiple variants.

    Example presets:
    - "all": Sample uniformly from 16 variants
    - "single_feat": Sample uniformly between CVRP, OVRP, VRPB, VRPL, VRPTW (as done in Liu et al. 2024 (MTPOMO))
    - "single_feat_otw": Sample uniformly between CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW (as done in Zhou et al. 2024 (MVMoE))
    - "cvrp": Only CVRP (similarly for other variants)

    Args:
        num_loc: Number of locations to generate
        min_loc: Minimum location value
        max_loc: Maximum location value
        loc_distribution: Distribution to sample locations from
        capacity: Vehicle capacity. If None, get value based on `get_vehicle_capacity`
        min_demand: Minimum demand value
        max_demand: Maximum demand value
        min_backhaul: Minimum backhaul value
        max_backhaul: Maximum backhaul value
        scale_demand: Scale demand values (by default, generate between 1 and 10)
        max_time: Maximum time window value (at depot)
        backhaul_ratio: Fraction of backhauls (e.g. 0.2 means 20% of nodes are backhaul)
        distance_limit: Distance limit
        speed: Speed of vehicle. Defaults to 1
        subsample: If False, we always sample all attributes (i.e., OVRPBLTW)
            If true, we use the
        **kwargs: Additional keyword arguments
    """
    def __init__(
            self,
            num_loc: int = 20, variant_preset=None, use_combinations=True, subsample=True,
            min_loc: float = 0.0, max_loc: float = 1.0,
            loc_distribution: Union[int, float, str, type, Callable] = Uniform,
            capacity: float = None, min_demand: int = 1, max_demand: int = 10,
            min_backhaul: int = 1, max_backhaul: int = 10,
            scale_demand: bool = True, max_time: float = 4.6, backhaul_ratio: float = 0.2,
            # distance_limit: float = 3.0, 
            distance_limit: float = 2.8,  # 2sqrt(2) ~= 2.8
            speed: float = 1.0,
            prob_open: float = 0.5, prob_time_window: float = 0.5,
            prob_limit: float = 0.5, prob_backhaul: float = 0.5,
            **kwargs,
    ) -> None:
        # Location distribution
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_sampler = get_sampler(loc_distribution, low=min_loc, high=max_loc, **kwargs)

        if capacity is None:
            capacity = get_vehicle_capacity(num_loc)
        self.capacity = capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_backhaul = min_backhaul
        self.max_backhaul = max_backhaul
        self.scale_demand = scale_demand
        self.backhaul_ratio = backhaul_ratio

        self.max_time = max_time
        self.distance_limit = distance_limit
        self.speed = speed

        assert not (subsample and (variant_preset is None)), (
            "Cannot use subsample if variant_preset is not specified. "
        )
        if variant_preset is not None:
            log.info(f"Using variant generation preset {variant_preset}")
            variant_probs = VARIANT_GENERATION_PRESETS.get(variant_preset)
            assert (
                    variant_probs is not None
            ), f"Variant generation preset {variant_preset} not found. \
                Available presets are {VARIANT_GENERATION_PRESETS.keys()} with probabilities {VARIANT_GENERATION_PRESETS.values()}"
        else:
            variant_probs = {
                "O": prob_open,
                "TW": prob_time_window,
                "L": prob_limit,
                "B": prob_backhaul,
            }
        # check probabilities
        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"
        self.variant_probs = variant_probs
        self.variant_preset = variant_preset
        if isinstance(variant_preset, str) and variant_preset != "all":
            log.warning(f"{variant_preset} selected. Will not use feature combination!")
            use_combinations = False
        self.use_combinations = use_combinations
        self.subsample = subsample

    def reset_n_loc(self, num_loc):
        self.num_loc = num_loc
        self.capacity = get_vehicle_capacity(num_loc)

    def reset_variant_preset(self, variant_preset):
        self.variant_preset = variant_preset
        variant_probs = VARIANT_GENERATION_PRESETS.get(variant_preset)
        self.variant_probs = variant_probs

    def __call__(self, batch_size) -> TensorDict:
        # _generate
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        # Locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2)).to('cuda')  # original device has being cuda, why I transfer it  (batch, num_loc, 2)
        depot = torch.FloatTensor(*batch_size, 1, 2).to('cuda').uniform_(self.min_loc, self.max_loc) # (batch, 1, 2)
        locs = torch.cat((depot, locs),dim=-2) # (batch, num_loc+1, 2)
        # Vehicle capacity (C, B) - applies to both linehaul and backhaul
        vehicle_capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)
        capacity_original = vehicle_capacity.clone()
        # linehaul demand / delivery (C) and backhaul / pickup demand (B)
        demand_linehaul, demand_backhaul = self.generate_demands(batch_size=batch_size, num_loc=self.num_loc)
        # add empty depot demands
        demand_linehaul = torch.cat([torch.zeros(size=(*batch_size, 1)), demand_linehaul], dim=1)
        demand_backhaul = torch.cat([torch.zeros(size=(*batch_size, 1)), demand_backhaul], dim=1)
        # Open (O)
        open_route = self.generate_open_route(shape=(*batch_size, 1))
        # Time windows (TW)
        speed = self.generate_speed(shape=(*batch_size, 1))
        time_windows, service_time = self.generate_time_windows(locs=locs,speed=speed,)
        # Distance limit (L)
        distance_limit = self.generate_distance_limit(shape=(*batch_size, 1), locs=locs)
        # scaling
        if self.scale_demand:
            demand_backhaul /= vehicle_capacity
            demand_linehaul /= vehicle_capacity
            vehicle_capacity /= vehicle_capacity
        # Put all variables together
        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,  # (C)
                "demand_linehaul": demand_linehaul,  # (B)
                "distance_limit": distance_limit,  # (L)
                "time_windows": time_windows,  # (TW)
                "service_time": service_time,  # (TW)
                "vehicle_capacity": vehicle_capacity,  # (C)
                "capacity_original": capacity_original,  # unscaled capacity (C)
                "open_route": open_route,  # (O)
                "speed": speed,  # common
            },batch_size=batch_size,
        )
        if self.subsample:
            # Subsample problems based on given instructions
            return self.subsample_problems(td)
        else:
            # Not subsampling problems, i.e. return tensordict with all attributes
            return td

    def subsample_problems(self, td):
        """Create subproblems starting from seed probabilities depending on their variant.
        If random seed sampled in [0, 1] in batch is greater than prob, remove the constraint
        thus, if prob high, it is less likely to remove the constraint (i.e. prob=0.9, 90% chance to keep constraint)
        """
        batch_size = td.batch_size[0]
        variant_probs = torch.tensor(list(self.variant_probs.values()))
        if self.use_combinations:
            # in a batch, multiple variants combinations can be picked
            keep_mask = torch.rand(batch_size, 4) >= variant_probs  # O, TW, L, B
        else:
            # in a batch, only a variant can be picked.
            # we assign a 0.5 prob to the last variant (which is normal cvrp)
            if self.variant_preset in list(
                    VARIANT_GENERATION_PRESETS.keys()
            ) and self.variant_preset not in (
                    "all",
                    "cvrp",
                    "single_feat",
                    "single_feat_otw",
            ):
                cvrp_prob = 0
            else:
                cvrp_prob = 0.5
            if self.variant_preset in ("all", "cvrp", "single_feat", "single_feat_otw"):
                indices = torch.distributions.Categorical(
                    torch.Tensor(list(self.variant_probs.values()) + [cvrp_prob])[
                        None
                    ].repeat(batch_size, 1)
                ).sample()
                if self.variant_preset == "single_feat_otw":
                    keep_mask = torch.zeros((batch_size, 6), dtype=torch.bool)
                    keep_mask[torch.arange(batch_size), indices] = True
                    # If keep_mask[:, 4] is True, make both keep_mask[:, 0] and keep_mask[:, 1] True
                    keep_mask[:, :2] |= keep_mask[:, 4:5]
                else:
                    keep_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
                    keep_mask[torch.arange(batch_size), indices] = True
            else:
                # if the variant is specified, we keep the attributes with probability > 0
                keep_mask = torch.zeros((batch_size, 4), dtype=torch.bool)
                indices = torch.nonzero(variant_probs).squeeze()
                keep_mask[:, indices] = True
        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_window(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])
        # td['problem_tag'] = torch.cat( (~keep_mask[:,0:1],keep_mask),dim=-1)  # C, O, TW, L, B
        # td['size_tag'] = torch.full_like(td['open_route'], (td['locs'].shape[1]-1)/2000, dtype=torch.float32)
        td['p_s_tag'] = torch.cat((~keep_mask[:, 0:1], keep_mask, torch.full_like(td['open_route'], (td['locs'].shape[1]-1)/2000, dtype=torch.float32)),dim=-1)
        return td

    @staticmethod
    def _default_open(td, remove):
        td["open_route"][remove] = False
        return td

    @staticmethod
    def _default_time_window(td, remove):
        default_tw = torch.zeros_like(td["time_windows"])
        default_tw[..., 1] = float("inf")
        td["time_windows"][remove] = default_tw[remove]
        td["service_time"][remove] = torch.zeros_like(td["service_time"][remove])
        return td

    @staticmethod
    def _default_distance_limit(td, remove):
        td["distance_limit"][remove] = float("inf")
        return td

    @staticmethod
    def _default_backhaul(td, remove):
        # by default, where there is a backhaul, linehaul is 0. therefore, we add backhaul to linehaul
        # and set backhaul to 0 where we want to remove backhaul
        td["demand_linehaul"][remove] = td["demand_linehaul"][remove] + td["demand_backhaul"][remove]
        td["demand_backhaul"][remove] = 0
        return td

    def generate_demands(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Classical lineahul demand / delivery from depot (C) and backhaul demand / pickup to depot (B) generation.
        Initialize the demand for nodes except the depot, which are added during reset.
        Demand sampling Following Kool et al. (2019), demands as integers between 1 and 10.
        Generates a slightly different distribution than using torch.randint.

        Returns:
            linehaul_demand: [B, N]
            backhaul_demand: [B, N]
        """
        linehaul_demand = (
                torch.FloatTensor(*batch_size, num_loc).to('cuda')
                .uniform_(self.min_demand - 1, self.max_demand - 1).int()+ 1
        ).float()
        # Backhaul demand sampling
        backhaul_demand = (
                torch.FloatTensor(*batch_size, num_loc).to('cuda')
                .uniform_(self.min_backhaul - 1, self.max_backhaul - 1).int()+ 1
        ).float()
        is_linehaul = torch.rand(*batch_size, num_loc) > self.backhaul_ratio
        backhaul_demand = ( backhaul_demand * ~is_linehaul)  # keep only values where they are not linehauls
        linehaul_demand = ( linehaul_demand * is_linehaul)
        return linehaul_demand, backhaul_demand

    def generate_time_windows(
            self,
            locs: torch.Tensor,
            speed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)
            speed: [B]

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """

        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1  # no depot

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length

        # Depot tw is 0, max_time
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),  # start
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),  # en
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        return time_windows, service_time  # [B, N+1, 2], [B, N+1]

    def generate_distance_limit(
        self, shape: Tuple[int, int], locs: torch.Tensor
        ) -> torch.Tensor:
        """Generates distance limits (L).
        The distance lower bound is dist_lower_bound = 2 * max(depot_to_location_distance),
        then the max can be max_lim = min(max_distance_limit, dist_lower_bound + EPS). Ensures feasible yet challenging
        constraints, with each instance having a unique, meaningful limit

        Returns:
            distance_limit: [B, 1]
        """
        max_dist = torch.max(torch.cdist(locs[:, 0:1], locs[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.distance_limit),
            dist_lower_bound + 1e-6,
        )
        # We need to sample from the `distribution` module to get the same distribution with a tensor as input
        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]


    def generate_open_route(self, shape: Tuple[int, int]):
        """Generate open route flags (O). Here we could have a sampler but we simply return True here so all
        routes are open. Afterwards, we subsample the problems.
        """
        return torch.ones(shape, dtype=torch.bool)

    def generate_speed(self, shape: Tuple[int, int]):
        """We simply generate the speed as constant here"""
        # in this version, the speed is constant but this class may be overridden
        return torch.full(shape, self.speed, dtype=torch.float32)

    @staticmethod
    def save_data(td: TensorDict, path, compress: bool = False):
        save_tensordict_to_npz(td, path)

    @staticmethod
    def print_presets():
        for key, value in VARIANT_GENERATION_PRESETS.items():
            print(f"{key}: {value}")

    @staticmethod
    def available_variants(*args, **kwargs):
        # remove 'all', 'single_feat' from the list
        return list(VARIANT_GENERATION_PRESETS.keys())[3:]


