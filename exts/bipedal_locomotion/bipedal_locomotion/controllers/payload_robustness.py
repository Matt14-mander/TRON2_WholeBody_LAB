"""Pure tensor rules for the first fixed-arm payload robustness experiment."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class RobustSettings:
    mass_range: tuple = (.25, .75)
    friction_scale_range: tuple = (.8, 1.2)
    max_force_n: float = 2.
    nominal_mass_kg: float = .5
    nominal_fraction: float = .4
    zero_probability: float = .25
    resampling_time_s: tuple = (2., 5.)
    warmup_s: float = .5
    ramp_s: float = .2
    material_buckets: int = 16

    def validate(self):
        for values in (self.mass_range, self.friction_scale_range, self.resampling_time_s):
            if len(values) != 2 or not all(math.isfinite(v) for v in values) or not 0 < values[0] <= values[1]:
                raise ValueError("Mass, friction scale and resampling intervals require positive ordered finite pairs")
        if (not all(math.isfinite(v) for v in (self.max_force_n, self.nominal_mass_kg,
                self.nominal_fraction, self.zero_probability, self.warmup_s, self.ramp_s))
                or self.max_force_n < 0 or self.nominal_mass_kg <= 0 or self.warmup_s < 0
                or self.ramp_s <= 0 or not 0 <= self.nominal_fraction <= 1
                or not 0 <= self.zero_probability <= 1 or not 1 <= self.material_buckets <= 64):
            raise ValueError("Invalid payload robustness settings")
        return self


def nominal_mask(count, fraction, device):
    mask = torch.arange(count, device=device) < int(count * fraction)
    return mask[torch.randperm(count, device=device)]


def sample_force(count, maximum, zero_probability, device):
    angle = torch.rand(count, device=device) * (2 * math.pi)
    radius = torch.rand(count, device=device).sqrt() * maximum
    radius *= torch.rand(count, device=device) >= zero_probability
    return torch.stack((radius * angle.cos(), radius * angle.sin(), torch.zeros_like(radius)), -1)


def scale_payload_inertia(default_inertia, mass, default_mass):
    """Always scale the authored inertia, never the previously randomized value."""
    return default_inertia * (mass / default_mass)[:, None]


class ForceSchedule:
    """Episode-local clock; uniform disk forces with continuous bounded ramps."""
    def __init__(self, nominal, settings):
        self.settings = settings.validate()
        self.nominal = nominal.clone()
        n, device = len(nominal), nominal.device
        self.current = torch.zeros(n, 3, device=device)
        self.target = torch.zeros_like(self.current)
        self.start = torch.zeros_like(self.current)
        self.ttl = torch.zeros(n, device=device)
        self.elapsed = torch.zeros(n, device=device)
        self.reset()

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(len(self.nominal), device=self.nominal.device)
        self.current[ids] = self.target[ids] = self.start[ids] = 0.
        self.elapsed[ids] = 0.
        self.ttl[ids] = torch.where(self.nominal[ids], float("inf"), self.settings.warmup_s)

    def advance(self, dt):
        self.ttl -= dt
        self.elapsed += dt
        ids = (self.ttl <= 0).nonzero(as_tuple=False).flatten()
        if len(ids):
            self.start[ids] = self.current[ids]
            self.target[ids] = sample_force(len(ids), self.settings.max_force_n,
                                           self.settings.zero_probability, self.current.device)
            self.elapsed[ids] = 0.
            low, high = self.settings.resampling_time_s
            self.ttl[ids] = low + torch.rand(len(ids), device=self.current.device) * (high - low)
        alpha = (self.elapsed / self.settings.ramp_s).clamp(0., 1.)[:, None]
        self.current.copy_(self.start + alpha * (self.target - self.start))
        return self.current
