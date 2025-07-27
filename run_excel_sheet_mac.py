import os
import time
import xlwings as xw
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict


def evaluate_excel_sheet(
    wb, inputs, cell_refs, output_vars=None, sheet=1
):
    """Inserts input variable values into specified cells,
    recalculates the sheet, and then reads the values of the
    remaining specified cells and returns them as a dictionary.

    The names and cell locations of the input and output
    variables are specified in the cell_refs dictionary.

    Args:
        wb: An open Excel workbook (xlwings.Book object).
        inputs (dict): values to be assigned to input variables.
            E.g. inputs = {'x': [0.0, 1.0]}
        sheet (int): Sheet number (1-based).
    
    Returns:
        outputs (dict): The cell values for all variables except the
            input variable(s) after the sheet was recalculated.

    Example

    To set up a function evaluation, f(x), that takes two inputs,
    x[0] and x[1], has a constraint function, g(x) <= 0, and lower
    and upper bounds on x, x_lb and x_ub, the Excel sheet could
    look something like this:

         | A       B       C       D
    -----|----------------------------------
       1 |                                  
       2 |         name    My Problem       
       3 |         f(x)    3.0              
       4 |         x       1.0     2.0      
       5 |         g(x)    10.0             
       6 |         x_lb    0.0     0.0      
       7 |         x_ub    5.0     5.0      

    The cell references for this example would be:

    cell_refs = {
        'name': ('B2', 'C2'),
        'f(x)': ('B3', 'C3'),
        'x': ('B4', ['C4', 'D4']),
        'g(x)': ('B5', 'C5'),
        'x_lb': ('B6', ['C6', 'D6']),
        'x_ub': ('B7', ['C7', 'D7'])
    }

    Notes:

    The reason the name fields are included is that the algorithm 
    checks that the text in the specified cells matches the variables 
    names to reduce the risk of referencing errors.

    To avoid unnecessary re-calculation, set the Excel calculation 
    option to 'manual'. Recalculation will be automatically triggered 
    when this function is called.
    """

    # Get the worksheet - xlwings uses 1-based indexing for sheets
    ws = wb.sheets[sheet - 1]  # xlwings uses 0-based indexing internally

    # Copy input variable values to specified cells
    for (var_name, values) in inputs.items():
        name_cell, value_cells = cell_refs[var_name]
        
        # Check variable name matches
        name_cell_value = ws.range(name_cell).value
        assert name_cell_value == var_name, f"variable name mismatch: expected {var_name}, got {name_cell_value}"
        
        # Set values
        try:
            len(values)
        except TypeError:
            # Single value
            ws.range(value_cells).value = values
        else:
            # Multiple values (list/array)
            if isinstance(value_cells, list):
                for cell_ref, value in zip(value_cells, values):
                    ws.range(cell_ref).value = value
            else:
                # If value_cells is a range like "C4:D4"
                ws.range(value_cells).value = values

    # Force Excel to recalculate all formulas
    wb.app.calculate()

    # If not specified, read all variable values except the inputs
    if output_vars is None:
        output_vars = set(cell_refs.keys()) - set(inputs.keys())

    outputs = {}
    for var_name in output_vars:
        name_cell, value_cells = cell_refs[var_name]

        # Verify variable name
        name_cell_value = ws.range(name_cell).value
        assert name_cell_value == var_name, f"variable name mismatch: expected {var_name}, got {name_cell_value}"

        # Get values
        if isinstance(value_cells, str):
            # Single cell
            value = ws.range(value_cells).value
        else:
            # Multiple cells - return as list
            value = [ws.range(cell_ref).value for cell_ref in value_cells]

        outputs[var_name] = value

    return outputs


def test_on_Toy1DProblem():

    # Path to your Excel file
    excel_sheets_dir = "excel_sheets"
    filename = "Toy-1D-Problem.xlsx"
    cell_refs = {
        'name': ('B2', 'C2'),
        'f(x)': ('B3', 'C3'),
        'x': ('B4', 'C4'),
        'x_lb': ('B5', 'C5'),
        'x_ub': ('B6', 'C6')
    }

    file_path = os.path.join(os.getcwd(), excel_sheets_dir, filename)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    # Open the workbook with xlwings
    try:
        wb = xw.Book(file_path)
        print(f"Successfully opened: {filename}")
    except Exception as e:
        print(f"Error opening Excel file: {e}")
        raise

    try:
        # Check the problem parameters
        inputs = {'x': 0.0}
        outputs = evaluate_excel_sheet(
            wb, inputs, cell_refs, output_vars=['name', 'x_lb', 'x_ub']
        )
        print("Test: ", outputs['name'])
        assert outputs['name'] == 'Toy1DProblem'
        assert outputs['x_lb'] == -5
        assert outputs['x_ub'] == 5

        x_values = np.linspace(-5, 5, 11)
        f_eval = []
        t0 = time.time()
        timings = []
        print("Evaluating excel sheet...")
        for i, x in tqdm(enumerate(x_values)):
            inputs = {'x': x}
            outputs = evaluate_excel_sheet(wb, inputs, cell_refs)
            f_eval.append(outputs['f(x)'])
            timings.append(time.time() - t0)

        # Print results and timings
        results_summary = pd.DataFrame(
            {'Time (seconds)': timings, 'x': x_values, 'f(x)': f_eval}
        )
        print(results_summary)

    finally:
        wb.save()
        wb.close()


def test_on_Toy2DProblemConstraint():

    # Path to your Excel file
    excel_sheets_dir = "excel_sheets"
    filename = "Toy-2D-Problem-Constraint.xlsx"
    cell_refs = {
        'name': ('B2', 'C2'),
        'f(x)': ('B3', 'C3'),
        'x': ('B4', ['C4', 'D4']),
        'g(x)': ('B5', 'C5'),
        'x_lb': ('B6', ['C6', 'D6']),
        'x_ub': ('B7', ['C7', 'D7'])
    }

    file_path = os.path.join(os.getcwd(), excel_sheets_dir, filename)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    # Open the workbook with xlwings
    try:
        wb = xw.Book(file_path)
        print(f"Successfully opened: {filename}")
    except Exception as e:
        print(f"Error opening Excel file: {e}")
        raise

    try:
        # Check the problem parameters
        inputs = {'x': [0.0, 0.0]}
        outputs = evaluate_excel_sheet(
            wb, inputs, cell_refs, output_vars=['name', 'x_lb', 'x_ub']
        )
        print("Test: ", outputs['name'])
        assert outputs['name'] == 'Toy2DProblemConstraint'
        assert outputs['x_lb'] == [-5, -5]
        assert outputs['x_ub'] == [5, 5]

        x1_values = np.linspace(-5, 5, 11)
        x2_values = np.linspace(-5, 5, 11)
        n_iters = x1_values.shape[0] * x2_values.shape[0]
        results = defaultdict(list)
        t0 = time.time()
        print("Evaluating excel sheet...")
        for x1, x2 in tqdm(product(x1_values, x2_values), total=n_iters):
            inputs = {'x': [x1, x2]}
            outputs = evaluate_excel_sheet(
                wb, inputs, cell_refs, output_vars=['f(x)', 'g(x)']
            )
            results['x1'].append(x1)
            results['x2'].append(x2)
            results['f_eval'].append(outputs['f(x)'])
            results['g_eval'].append(outputs['g(x)'])
            results['timings'].append(time.time() - t0)

        # Print results and timings
        results_summary = pd.DataFrame(results)
        print(results_summary)

    finally:
        wb.save()
        wb.close()


if __name__ == "__main__":
    test_on_Toy1DProblem()
    test_on_Toy2DProblemConstraint()