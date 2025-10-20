# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Higher-order Constraint with linear, quadratic, and higher-order terms."""

from typing import Any, Dict, List, Optional, Tuple, Union

from numpy import ndarray
from scipy.sparse import spmatrix

from .constraint import Constraint, ConstraintSense
from .higher_order_expression import HigherOrderExpression
from .linear_expression import LinearExpression
from .quadratic_expression import QuadraticExpression

CoeffLike = Union[ndarray, Dict[Tuple[Union[int, str]], float], List]


class HigherOrderConstraint(Constraint):
    """Constraint of the form:
    e.g. linear(x) + x^T Q x + sum_{k>=3}  sum_{|t|=k} C_k[t] * prod_{i in t} x[i]  `sense` `rhs`
    where `sense` is one of the ConstraintSense values (e.g., LE, <=) and `rhs` is a float.


    Supports both a single higher-order term (order+coeffs) and multiple via
    higher_orders={k: coeffs}.
    """

    Sense = ConstraintSense  # duplicated for Sphinx compatibility

    def __init__(
        self,
        optimization_problem: Any,
        name: str,
        # linear/quadratic
        linear: Optional[
            Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]]
        ] = None,
        quadratic: Optional[
            Union[
                ndarray,
                spmatrix,
                List[List[float]],
                Dict[Tuple[Union[int, str], Union[int, str]], float],
            ]
        ] = None,
        # higher-order
        higher_order: Optional[Dict[int, CoeffLike]] = None,
        sense: ConstraintSense = ConstraintSense.LE,
        rhs: float = 0.0,
    ) -> None:
        """Construct a higher-order constraint with linear, quadratic, and optional higher-order
        parts.

        Args:
            optimization_problem: The optimization problem this constraint belongs to.
            name: The name of the constraint.
            linear: The coefficients for the linear part of the constraint.
            quadratic: The coefficients for the quadratic part of the constraint.
            higher_order: A single higher-order expression or a dictionary of {order: coeffs}
                for multiple orders (k>=3).
            sense: The sense of the constraint (e.g., LE, <=).
            rhs: The right-hand-side value of the constraint.
        """
        super().__init__(optimization_problem, name, sense, rhs)

        self._linear = LinearExpression(optimization_problem, {} if linear is None else linear)
        self._quadratic = QuadraticExpression(
            optimization_problem, {} if quadratic is None else quadratic
        )

        # Store multiple higher-order expressions keyed by order (k>=3)
        if higher_order is None:
            self._higher_order: Dict[int, HigherOrderExpression] = {}
        else:
            self.higher_order = higher_order

    # --- properties ---
    @property
    def linear(self) -> LinearExpression:
        """Returns the linear expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side linear expression.
        """
        return self._linear

    @linear.setter
    def linear(self, linear):
        """Sets the linear expression corresponding to the left-hand-side of the constraint.

        Args:
            linear: The linear coefficients of the left-hand-side.
        """
        self._linear = LinearExpression(self.optimization_problem, linear)

    @property
    def quadratic(self) -> QuadraticExpression:
        """Returns the quadratic expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side quadratic expression.
        """
        return self._quadratic

    @quadratic.setter
    def quadratic(self, quadratic):
        """Sets the quadratic expression corresponding to the left-hand-side of the constraint.

        Args:
            quadratic: The quadratic coefficients of the left-hand-side.
        """
        self._quadratic = QuadraticExpression(self.optimization_problem, quadratic)

    @property
    def higher_order(self) -> Dict[int, HigherOrderExpression]:
        """Return a shallow copy of {order: HigherOrderExpression}.

        Returns:
            A dictionary mapping order (k>=3) to HigherOrderExpression.
        """
        return dict(self._higher_order)

    @higher_order.setter
    def higher_order(
        self,
        higher_order: Union[Dict[int, CoeffLike]],
    ) -> None:
        """Sets the higher-order expressions.

        Args:
            higher_order: A dictionary of {order: HigherOrderExpression} for multiple orders.
        """
        self._higher_order = {}

        for k, coeffs in higher_order.items():
            self._higher_order[k] = HigherOrderExpression(self.optimization_problem, coeffs)

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate the left-hand-side of the constraint.

        Args:
            x: The values of the variables to be evaluated.
        Returns:
            The left-hand-side of the constraint given the variable values.
        """
        val = self.linear.evaluate(x) + self.quadratic.evaluate(x)
        for expr in self._higher_order.values():
            val += expr.evaluate(x)
        return val

    def __repr__(self):
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        lhs = expr2str(
            linear=self.linear,
            quadratic=self.quadratic,
            higher_order=self._higher_order,
            truncate=DEFAULT_TRUNCATE,
        )
        return f"<{self.__class__.__name__}: {lhs} {self.sense.label} {self.rhs} '{self.name}'>"

    def __str__(self):
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        lhs = expr2str(
            linear=self.linear,
            quadratic=self.quadratic,
            higher_order=self._higher_order,
        )
        return f"{lhs} {self.sense.label} {self.rhs} '{self.name}'"
