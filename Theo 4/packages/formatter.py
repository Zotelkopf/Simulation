"""
Custom numeric and sqrt-aware formatting for NumPy and IPython display.

Designed for Jupyter notebooks running inside VS Code.
"""

import numpy as np
import matplotlib as mpl
from sympy import Expr, Integer, Matrix, N, latex, nsimplify, I, pi, E
from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell

# -----------------------
# Configurable parameters
# -----------------------
ZERO_TOL = 1e-15            # absolute threshold -> treat |x| < ZERO_TOL as zero
MAX_DENOM = 10_000          # max denominator to accept when nsimplify returns a Rational
SIG_DIGITS = 4              # default significant digits for fallback formatting
ALLOWED_SYMBOLIC = [pi, E]  # things we accept as "common" symbolic parts

# ------------------------------------------------------------
# CSS / Environment Initialization
# ------------------------------------------------------------
def _apply_vscode_css() -> None:
    """
    Injects CSS to improve widget/output display in VS Code.
    """

    css = """
    <style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        .jp-OutputArea-output {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }
    </style>
    """
    display(HTML(css))

# ------------------------------------------------------------
# NumPy / IPython Formatter Initialization
# ------------------------------------------------------------
def _apply_nsimplify():
    def _nsimplify_improved(
        value: complex | float | int, 
        zero_tol: float = ZERO_TOL, 
        max_denom: int = MAX_DENOM, 
        sig_digits: int = SIG_DIGITS, 
        allowed_symbolic: list[Expr] = ALLOWED_SYMBOLIC
    ) -> Expr:
        '''
        Improved version of sympy.nsimplify that:
        '''
        if np.iscomplexobj(value):
            real = _nsimplify_improved(value.real, zero_tol, max_denom, sig_digits, allowed_symbolic)
            imag = _nsimplify_improved(value.imag, zero_tol, max_denom, sig_digits, allowed_symbolic)
            if imag == 0:
                return real
            if real == 0:
                return imag
            return real + imag * I
        
        if abs(value) < zero_tol:
            return Integer(0)
        
        try:
            candidate = nsimplify(value, allowed_symbolic)
        except Exception:
            return N(value, sig_digits)
        
        try:
            if abs(float(N(candidate)) - value) > zero_tol:
                return N(value, sig_digits)
        except Exception:
            return N(value, sig_digits)
        
        if candidate.is_Rational and candidate.q > max_denom:
            return N(value, sig_digits)
        
        try:
            if not candidate.has(*ALLOWED_SYMBOLIC) and not candidate.is_Rational:
                return N(value, sig_digits)
        except Exception:
            return N(value, sig_digits)
        
        return candidate

    def _nsimplify_matrix(array: np.ndarray) -> str | None:
        if array.ndim > 2:    
            return None
        try:
            return latex(Matrix(array).applyfunc(lambda x: _nsimplify_improved(complex(x))))
        except Exception:
            return None
    
    def _nsimplify_scalar(value: complex | float | int) -> str | None:
        try:
            return latex(_nsimplify_improved(value))
        except Exception:
            return None
    
    shell = InteractiveShell.instance()
    fmt = shell.display_formatter.formatters['text/latex'] # type: ignore
    
    fmt.for_type(np.ndarray, lambda obj: f"${_nsimplify_matrix(obj)}$")

    for type in [np.complexfloating, np.floating, np.integer]:
        fmt.for_type(type, lambda obj: f"${_nsimplify_scalar(obj)}$")

# ------------------------------------------------------------
# Main Initialization
# ------------------------------------------------------------
def init_formatters() -> None:
    """
    Configure matplotlib, CSS, NumPy printing, and IPython scalar display.
    """
    
    mpl.rcParams["figure.dpi"] = 120
    _apply_vscode_css()
    _apply_nsimplify()
