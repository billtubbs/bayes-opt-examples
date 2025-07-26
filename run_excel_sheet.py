import win32com.client as win32
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


def evaluate_excel_sheet(wb, x, x_rows=[4], f_row=5, sheet=1):
    """Opens an excel spreadsheet, inserts values in input cells,
    executes the sheet, and reads the result from the output cell.

    The requirement is that the variables names should be in column
    2 and the variable values in column 3.

    For example, for a function that takes two inputs, x[0] and x[1]
    the Excel sheet should look something like this:

         | A       B       C
    -----|--------------------------
       1 |                          
       2 |         Function name    
       3 |                          
       4 |         x0      1.0      
       5 |         x1      2.0      
       6 |         f(x)    3.0      

    Args:
        wb (win32com.gen_py.Workbook): An open Excel workbook.
        x (list): values of x.
        x_rows (list): rows where values of x should be inserted.
        f_row (int): row where value of f(x) will be calculated.
        sheet (int): Sheet number.
    
    Returns:
        f_value (float): Calculated value of function.
    """

    ws = wb.Sheets(sheet)
    for i, (row, xi) in enumerate(zip(x_rows, x)):
        # Find 'xi' cell and set the adjacent value
        if ws.Cells(row, 2).Value == 'x0':
            ws.Cells(row, 3).Value = xi
        else:
            raise ValueError(f"cell 'x{i}' not found")

    # Force recalculation
    excel.CalculateUntilAsyncQueriesDone()
    wb.RefreshAll()
    excel.CalculateFullRebuild()

    # Find 'f(x)' and read the adjacent value
    if ws.Cells(f_row, 2).Value == 'f(x)':
        f_value = ws.Cells(f_row, 3).Value
    else:
        raise ValueError("cell 'f(x)' not found")

    return f_value


if __name__ == "__main__":

    # Path to your Excel file
    excel_sheets_dir = "excel_sheets"
    filename = "Toy-1D-Problem.xlsx"  
    file_path = os.path.join(os.getcwd(), excel_sheets_dir, filename)

    # Launch Excel
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    excel.Visible = False  # Set to True if you want to see Excel open

    # Open the workbook
    wb = excel.Workbooks.Open(file_path)

    x_values = np.linspace(-5, 5, 11)
    f_eval = []
    t0 = time.time()
    timings = []
    print("Evaluating excel sheet...")
    for i, x in tqdm(enumerate(x_values)):
        f_eval.append(evaluate_excel_sheet(wb, [x]))
        timings.append(time.time() - t0)

    # Save and close
    wb.Save()
    wb.Close()
    excel.Quit()

    # Print results and timings
    results_summary = pd.DataFrame(
        {'Time (seconds)': timings, 'x': x_values, 'f(x)': f_eval}
    )
    print(results_summary)
