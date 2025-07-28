import os
import time
import pytest
import xlwings as xw
import numpy as np
import pandas as pd
from excel_tools.run_excel_sheet_mac import evaluate_excel_sheet
from pandas.testing import assert_frame_equal
from tqdm import tqdm
from itertools import product
from collections import defaultdict


@pytest.fixture()
def test_data_dir():
    return 'test/test_data'


@pytest.fixture()
def excel_sheets_dir():
    return 'excel_sheets'


def test_on_Toy1DProblem(excel_sheets_dir, test_data_dir):

    # Path to your Excel file
    filename = "Toy-1D-Problem.xlsx"
    cell_refs = {
        'name': ('B2', 'C2'),
        'f(x)': ('B3', 'C3'),
        'x': ('B4', 'C4'),
        'x_lb': ('B5', 'C5'),
        'x_ub': ('B6', 'C6')
    }

    file_path = os.path.join(excel_sheets_dir, filename)

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
        print("Test name: ", outputs['name'])
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

        # Check results match data on file
        filename = 'Toy1DProblem.csv'
        expected_results = pd.read_csv(
            os.path.join(test_data_dir, filename), index_col=0
        )
        assert_frame_equal(results_summary[['x', 'f(x)']], expected_results)

    finally:
        wb.save()
        wb.close()


def test_on_Toy2DProblemConstraint(excel_sheets_dir, test_data_dir):

    # Path to your Excel file
    filename = "Toy-2D-Problem-Constraint.xlsx"
    cell_refs = {
        'name': ('B2', 'C2'),
        'f(x)': ('B3', 'C3'),
        'x': ('B4', ('C4', 'D4')),
        'g(x)': ('B5', 'C5'),
        'x_lb': ('B6', 'C6:D6'),  # alternative way to define range
        'x_ub': ('B7', 'C7:D7')
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

        # Check results match data on file
        filename = 'Toy2DProblemConstraint.csv'
        expected_results = pd.read_csv(
            os.path.join(test_data_dir, filename), index_col=0
        )
        assert_frame_equal(
            results_summary[['x1', 'x2', 'f_eval', 'g_eval']],
            expected_results
        )

    finally:
        wb.save()
        wb.close()
