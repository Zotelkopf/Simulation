"""
Custom numeric and sqrt-aware formatting for NumPy and IPython display.

Designed for Jupyter notebooks running inside VS Code.
"""

import numpy as np
import matplotlib as mpl
from sympy import Expr, Symbol, Integer, Matrix, N, latex, nsimplify, I, pi, E
from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell

# -----------------------
# Configurable parameters
# -----------------------
ZERO_TOL = 1e-15            # absolute threshold -> treat |x| < ZERO_TOL as zero
MAX_DENOM = 10_000          # max denominator to accept when nsimplify returns a Rational
SIG_DIGITS = 4              # default significant digits for fallback formatting
ALLOWED_SYMBOLIC = [pi, E]  # things we accept as "common" symbolic parts
TRUNCATE_TO = (6, 6)        # max rows/cols before truncating display with ellipsis
EMPTY_STR = Symbol(r"")     # symbol for empty string
V_ROWS = Symbol(r"\vdots")  # symbol for vertical ellipsis
H_ROWS = Symbol(r"\cdots")  # symbol for horizontal ellipsis
D_ROWS = Symbol(r"\ddots")  # symbol for diagonal ellipsis

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
# Formatter Initialization
# ------------------------------------------------------------
def _apply_nsimplify():
    def _nsimplify_improved(
        value: complex | float | int, 
        zero_tol: float = ZERO_TOL, 
        max_denom: int = MAX_DENOM, 
        sig_digits: int = SIG_DIGITS, 
        allowed_symbolic: list[Expr] = ALLOWED_SYMBOLIC
    ) -> Expr:
        # --- Complex case --------------------------------------------------
        if np.iscomplexobj(value):
            real = _nsimplify_improved(value.real, zero_tol, max_denom, sig_digits, allowed_symbolic)
            imag = _nsimplify_improved(value.imag, zero_tol, max_denom, sig_digits, allowed_symbolic)
            if imag == 0:
                return real
            if real == 0:
                return imag
            return real + imag * I
        
        # --- Near-zero case ------------------------------------------------
        if abs(value) < zero_tol:
            return Integer(0)
        
        # --- Real case -----------------------------------------------------
        try:
            candidate = nsimplify(value, allowed_symbolic)
            if abs(float(N(candidate)) - value) > zero_tol:
                return N(value, sig_digits)
            if candidate.is_Rational and candidate.q > max_denom:
                return N(value, sig_digits)
            if candidate.as_numer_denom()[1].has(*allowed_symbolic):
                return N(value, sig_digits)
            if not candidate.has(*allowed_symbolic) and not candidate.is_Rational:
                return N(value, sig_digits)
        except Exception:
            return N(value, sig_digits)
        
        return candidate

    def _nsimplify_matrix(
        array: np.ndarray,
        truncate_to: tuple[int, int] = TRUNCATE_TO
    ) -> str | None:
        if array.size == 0:
            return None
        if array.dtype.kind in {'U', 'S', 'O'}:
            return None
        if array.ndim > 2:    
            return None
        
        # --- 0D case -------------------------------------------------------
        if array.ndim == 0:
            return _nsimplify_scalar(array.item())
        
        # --- Prepare truncation indices ------------------------------------
        max_cols, max_rows = truncate_to
        rows = [i for i in range(max_rows - 1)] + [-1]
        cols = [i for i in range(max_cols - 1)] + [-1]

        # --- 1D truncated case ---------------------------------------------
        if array.ndim == 1 and array.size > max_rows:
            try:
                M = Matrix(array[np.ix_(rows)]).applyfunc(lambda x: _nsimplify_improved(complex(x)))
                M = M.row_insert(max_rows - 1, Matrix([[V_ROWS]]))
                return latex(M)
            except Exception:
                return None

        # --- 2D truncated case ---------------------------------------------
        # TODO: Handle non-square matrices with many rows or columns
        if array.ndim == 2 and array.shape[0] > max_cols and array.shape[1] > max_rows:
            try:
                M = Matrix(array[np.ix_(rows, cols)]).applyfunc(lambda x: _nsimplify_improved(complex(x)))
                M = M.row_insert(max_rows - 1, Matrix([[EMPTY_STR, EMPTY_STR, V_ROWS, EMPTY_STR, EMPTY_STR, EMPTY_STR]]))
                M = M.col_insert(max_cols - 1, Matrix([[EMPTY_STR], [EMPTY_STR], [H_ROWS], [EMPTY_STR], [EMPTY_STR], [D_ROWS], [EMPTY_STR]]))
                return latex(M)
            except Exception:
                return None
        
        # --- Non-truncated case ---------------------------------------------
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
