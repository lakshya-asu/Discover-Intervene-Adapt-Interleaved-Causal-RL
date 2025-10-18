import numpy as np

from dia.evgs import EVGS
from dia.types import Subgoal, Predicate


def test_predicates():
    evgs = EVGS(var_names=["a", "b"], obs_to_vars=lambda obs: np.array(obs))
    x0 = np.array([0.0, 1.0])
    x1 = np.array([0.5, 0.5])

    assert EVGS.predicate_holds(x0, x1, Subgoal(0, Predicate.UP))
    assert EVGS.predicate_holds(x0, x1, Subgoal(1, Predicate.DOWN))
    assert EVGS.predicate_holds(x0, x0, Subgoal(0, Predicate.EQUAL, value=0.0))
    assert EVGS.predicate_holds(x1, x1, Subgoal(1, Predicate.REACH, value=0.5))
