from .curriculum import (
    ZONES_CURRICULUM, LETTER_CURRICULUM, FLATWORLD_CURRICULUM, FLATWORLD_BIG_CURRICULUM,
    ZONES_PLANNING_CURRICULUM, HARD_OPTIMALITY_CURRICULUM, ZONES_TWOSTEP_CURRICULUM,
    ZONES_MIXED_CURRICULUM,
)
from .curriculum_sampler import CurriculumSampler

curricula = {
    # Standard curricula
    'PointLtl2-v0': ZONES_CURRICULUM,              # V0: baseline (starts with 1-step)
    'PointLtl2-v0.twostep': ZONES_TWOSTEP_CURRICULUM,   # V1: starts with 2-step only
    'PointLtl2-v0.mixed': ZONES_MIXED_CURRICULUM,       # V2: 75% 2-step + 25% 1-step
    # Specialized curricula
    'PointLtl2-planning-v0': ZONES_PLANNING_CURRICULUM,  # Extended with planning stages
    'PointLtl2-v0.hardmix': HARD_OPTIMALITY_CURRICULUM,  # Hard optimality training
    # Other environments
    'LetterEnv-v0': LETTER_CURRICULUM,
    'FlatWorld-v0': FLATWORLD_CURRICULUM,
    'FlatWorld-big-v0': FLATWORLD_BIG_CURRICULUM,
}
