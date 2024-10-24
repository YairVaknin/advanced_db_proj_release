# Interactive QueryProcessor Shell

## Overview

The `interactive_loader.py` script provides an interactive IPython shell for experimenting with the `QueryProcessor` class. It allows you to:

- Load datasets into the `QueryProcessor`.
- Perform relational algebra operations with provenance tracking.
- Calculate probabilities of query results in probabilistic databases.
- Experiment with different semiring implementations and semantics (`set` or `bag`).

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages:
  - `pandas`
  - `numpy`
  - `IPython`
  - `sympy` (for probability calculations)
  - `pytest` (for running tests, if needed)
  - `mypy` (for type checking, if needed)

### Installation

Install the required packages using `pip`:

```bash
pip install pandas numpy IPython sympy pytest mypy
```

### Usage

Run the `interactive_loader.py` script from the command line:

```bash
python interactive_loader.py [OPTIONS]
```

## Command-Line Arguments

The script supports the following command-line arguments:

- `--mode`: Mode of operation. Options are:
  - `test`: Loads small toy tables for testing purposes.
  - `default`: Loads CSV files from the `olympic_historical_dataset` folder (default behavior).
  - `custom`: Loads CSV files from a custom directory specified by the `--path` argument.
- `--path`: The directory path containing CSV files (used in `custom` mode).
- `--semiring`: The semiring to use for provenance calculations. Options are:
  - `NaturalNumbersSemiring`
  - `PolynomialSemiring` (default)
  - `BooleanSemiring`
  - `BooleanPolynomialSemiring`
  - `TropicalCostSemiring`
  - `AccessControlSemiring`
- `--semantics`: The semantics to use. Options are:
  - `set` (default)
  - `bag`
- `--load_file`: (Optional) The path to a saved `QueryProcessor` instance to load. If provided, this bypasses the normal data-loading behavior and initializes the `QueryProcessor` with the saved data and state.

### Example Usage

- Load default datasets with the default semiring and semantics:

  ```bash
  python interactive_loader.py
  ```

- Load test data with the `BooleanPolynomialSemiring` and `bag` semantics:

  ```bash
  python interactive_loader.py --mode test --semiring BooleanPolynomialSemiring --semantics bag
  ```

- Load custom CSV files from a directory:

  ```bash
  python interactive_loader.py --mode custom --path /path/to/csv/files
  ```

- Load a saved `QueryProcessor` instance from a file:

  ```bash
  python interactive_loader.py --load_file /path/to/saved_instance.pkl
  ```

## Interactive Shell Usage

After running the script, an IPython shell will start with the `QueryProcessor` instance loaded and ready for use.

### Available Variables and Methods

- `qp`: The `QueryProcessor` instance.
- `qp.tables`: A dictionary containing the loaded tables as pandas DataFrames.
- `qp.get_available_tables()`: A method to display available table names.
- `qp.save_qp(filepath)`: Saves the current `QueryProcessor` instance to a file.
- `QueryProcessor.load_qp(filepath)`: Loads a `QueryProcessor` instance from a file (class method).
- Other imported modules: `pandas` (`pd`).

### Relational Algebra Operations with Provenance Tracking

The `QueryProcessor` provides methods to perform basic relational algebra operations with provenance tracking:

- **Selection**: `qp.select(table, condition)`
- **Projection**: `qp.project(table, columns)`
- **Rename**: `qp.rename(table, columns)`
- **Join**: `qp.join(left, right, on, how)`
- **Union**: `qp.union(table1, table2)`
- **Intersection**: `qp.intersection(table1, table2)`
- **Difference**: `qp.difference(table1, table2)`

These operations can be combined to form more complex queries.

The `table` arguments in these operations can either be the name of a table (a string that corresponds to a table in `qp.tables`) or a pandas `DataFrame` object. This flexibility allows you to easily mix table names and query results in your operations.

### Query Examples

#### Selection

```python
# Select students majoring in 'Physics'
physics_students = qp.select(table='students', condition="major == 'Physics'")
```

#### Projection

```python
# Project the 'name' and 'age' columns from the students table
student_names_ages = qp.project(table='students', columns=['name', 'age'])
```

#### Rename

```python
# Rename 'name' to 'student_name' and 'age' to 'student_age' in the students table
renamed_students = qp.rename(table='students', columns={'name': 'student_name', 'age': 'student_age'})
```

#### Join

```python
# Join students and enrollments on 'student_id'
students_enrollments = qp.join(left='students', right='enrollments', on=['student_id'])
```

#### Union

```python
# Union of two tables
union_result = qp.union(table1='table1', table2='table2')
```

#### Intersection

```python
# Intersection of two tables
intersection_result = qp.intersection(table1='table1', table2='table2')
```

#### Difference

```python
# Difference between two tables (rows in table1 not in table2)
difference_result = qp.difference(table1='table1', table2='table2')
```

### Calculating Probabilities

For semirings that support probability calculations (e.g., `PolynomialSemiring`, `BooleanPolynomialSemiring`), you can compute the probability of a provenance expression.

#### Probability Calculation Modes

- **Exact Computation**: Calculates the exact probability using a recursive approach.
- **Monte Carlo Sampling**: Estimates the probability using random sampling.

#### Probability Example

```python
# Get a provenance expression from a query result
provenance_expr = some_dataframe.iloc[0]['provenance']

# Compute the exact probability
probability = qp.compute_probability(provenance_expr, exact_computation=True)
print(f"Exact probability: {probability}")

# Compute the approximate probability using Monte Carlo sampling
probability_approx = qp.compute_probability(provenance_expr, exact_computation=False)
print(f"Approximate probability: {probability_approx}")
```

### Available Tables

Use `qp.get_available_tables()` to see the list of loaded tables.

### Accessing Tables

Tables are stored as pandas DataFrames in `qp.tables`. You can access them using:

```python
students_df = qp.tables['students']
```

### Loading Additional Data

You can load additional DataFrames into the `QueryProcessor` using `qp.load_table()`.

```python
import pandas as pd

# Create a new DataFrame
new_data = pd.DataFrame({
    'id': [1, 2, 3],
    'value': ['A', 'B', 'C']
})

# Load it into the QueryProcessor
qp.load_table(df=new_data, table_name='new_table')
```

**Note**: When you are adding query results (e.g., selections, projections, joins) back into the `QueryProcessor`, you should use `qp.insert_table()` instead of `qp.load_table()`. This is because `insert_table` directly adds the result without assigning new probabilities or provenance, preserving the integrity of the query result’s provenance. In contrast, `load_table` assumes that the input table either has no provenance and probabilities or should be re-processed for provenance and probability assignment.
Also, all our tables are pandas DataFrames, so you can use all the DataFrame methods and attributes on them, such as adding new columns/rows, creation of new DataFrames, etc.

#### Example

```python
# Result from a query (e.g., projection)
query_result = qp.project(table='students', columns=['name', 'age'])

# Insert the result back into the QueryProcessor
qp.insert_table(table_name='projected_students', df=query_result)
```

If you attempt to use `load_table()` for a result DataFrame that already contains provenance and probabilities, these will be reprocessed unnecessarily, which may lead to an unexpected or inconsistent behavior.

## Supported Semirings

The following semirings are available:

- **NaturalNumbersSemiring**: Uses natural numbers with standard addition and multiplication.
- **PolynomialSemiring**: Represents provenance as polynomials with symbolic expressions.
- **BooleanSemiring**: Uses logical OR as addition and logical AND as multiplication.
- **BooleanPolynomialSemiring**: Represents boolean expressions using symbolic logic operations.
- **TropicalCostSemiring**: Used in optimization problems, with addition as minimum and multiplication as addition.
- **AccessControlSemiring**: For access control levels, with addition and multiplication defined accordingly.

## Semantics

You can choose between `set` and `bag` semantics:

- **Set Semantics**: Duplicate rows are eliminated, and provenance expressions are combined using semiring addition.
- **Bag Semantics**: Duplicate rows are retained, and provenance expressions reflect each individual tuple.

## Notes

- **Provenance**: Each row in the DataFrames managed by `QueryProcessor` has a `provenance` column containing the provenance expression. You can specify `provenance_col_name` if the table include a provenance column when invoking `load_table`, which will be converted into the semiring format.
- **Probabilities**: When loading tables, you can specify a `probability_col_name` if your data includes probabilities for each tuple. If not provided, random probabilities will be assigned.
- **Safety**: While using the interactive shell, runtime safety is not guaranteed. Avoid modifying or reassigning key attributes of the `QueryProcessor` instance (e.g., `qp.semiring`, `qp.semantics`) during the session to prevent unintended behavior.

## Example Sessions

### Test Session (Toy Data)

To start the interactive shell, run the following command:

```bash
python interactive_loader.py --mode test --semiring PolynomialSemiring --semantics set
```

This will load the test data and start the interactive shell with the specified semiring (`PolynomialSemiring`) and semantics (`set`). You can experiment with different semirings and database semantics by changing the respective arguments.

Here's an example of an interactive session using the shell:

```python
Test data loaded into QueryProcessor.
   ___                        ____                                           ____  _          _ _
  / _ \ _   _  ___ _ __ _   _|  _ \ _ __ ___   ___ ___  ___ ___  ___  _ __  / ___|| |__   ___| | |
 | | | | | | |/ _ \ '__| | | | |_) | '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__| \___ \| '_ \ / _ \ | |
 | |_| | |_| |  __/ |  | |_| |  __/| | | (_) | (_|  __/\__ \__ \ (_) | |     ___) | | | |  __/ | |
  \__\_\\__,_|\___|_|   \__, |_|   |_|  \___/ \___\___||___/___/\___/|_|    |____/|_| |_|\___|_|_|
                        |___/

Welcome to the Interactive QueryProcessor Shell!
You can access the QueryProcessor instance using the variable 'qp'.
Available tables are stored in 'qp.tables'.
Available table names:
 * students
 * courses
 * enrollments

In [1]: qp.get_available_tables()
Available tables:
 * students
 * courses
 * enrollments

In [2]: students_df = qp.tables['students']
   ...: students_df
Out[2]:
   student_id     name  age      major    provenance
0           1    Alice   20    Physics  X_students_0
1           2      Bob   22  Chemistry  X_students_1
2           3  Charlie   21    Physics  X_students_2

In [3]: # Select students majoring in Physics
   ...: physics_students = qp.select(table='students', condition="major == 'Physics'")
   ...: physics_students
Out[3]:
   student_id     name  age    major    provenance
0           1    Alice   20  Physics  X_students_0
1           3  Charlie   21  Physics  X_students_2

In [4]: # Join physics students with enrollments
   ...: physics_enrollments = qp.join(left=physics_students, right='enrollments', on=['student_id'])
   ...: physics_enrollments
Out[4]:
   student_id     name  age    major  course_id                    provenance
0           1    Alice   20  Physics        201  X_enrollments_0*X_students_0
1           3  Charlie   21  Physics        101  X_enrollments_2*X_students_2

In [5]: # Compute the probability of a result
   ...: provenance_expr = physics_enrollments.iloc[0]['provenance']
   ...: exact_probability = qp.compute_probability(provenance_expr, exact_computation=True)
   ...: sampled_probability = qp.compute_probability(provenance_expr, exact_computation=False)
   ...: print(f"Exact probability comutation of the first result: {exact_probability}")
Exact probability comutation of the first result: 0.792
   ...: print(f"Sampled probability comutation of the first result: {sampled_probability}")
Sampled probability comutation of the first result: 0.7873
   ...: physics_projection = qp.project(table=physics_enrollments, columns=["major"])
   ...: physics_projection
Out[...]:
     major                                         provenance
0  Physics  X_enrollments_0*X_students_0 + X_enrollments_2...
   ...: provenance_expr2 = physics_projection.iloc[0]['provenance']
   ...: provenance_expr2
X_enrollments_0*X_students_0 + X_enrollments_2*X_students_2
   ...: exact_probability2 = qp.compute_probability(provenance_expr2, exact_computation=True)
   ...: sampled_probability2 = qp.compute_probability(provenance_expr2, exact_computation=False)
   ...: print(f"Exact probability comutation of the first result: {exact_probability2}")
Exact probability comutation of the first result: 0.95112
   ...: print(f"Sampled probability comutation of the first result: {sampled_probability2}")
Sampled probability comutation of the first result: 0.953
```

### Default Session (Olympic Historical Dataset)

To start the interactive shell, run the following command:

```bash
python interactive_loader.py --semiring PolynomialSemiring --semantics set
```

```python
Loaded Olympics_Country.csv into table 'Olympics_Country' with assigned provenance and probabilities.
Loaded Olympics_Games.csv into table 'Olympics_Games' with assigned provenance and probabilities.
Loaded Olympic_Athlete_Bio.csv into table 'Olympic_Athlete_Bio' with assigned provenance and probabilities.
Loaded Olympic_Athlete_Event_Results.csv into table 'Olympic_Athlete_Event_Results' with assigned provenance and probabilities.
Loaded Olympic_Games_Medal_Tally.csv into table 'Olympic_Games_Medal_Tally' with assigned provenance and probabilities.
Loaded Olympic_Results.csv into table 'Olympic_Results' with assigned provenance and probabilities.
All CSV files from 'olympic_historical_dataset' have been loaded into the QueryProcessor.

   ___                        ____                                           ____  _          _ _
  / _ \ _   _  ___ _ __ _   _|  _ \ _ __ ___   ___ ___  ___ ___  ___  _ __  / ___|| |__   ___| | |
 | | | | | | |/ _ \ '__| | | | |_) | '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__| \___ \| '_ \ / _ \ | |
 | |_| | |_| |  __/ |  | |_| |  __/| | | (_) | (_|  __/\__ \__ \ (_) | |     ___) | | | |  __/ | |
  \__\_\\__,_|\___|_|   \__, |_|   |_|  \___/ \___\___||___/___/\___/|_|    |____/|_| |_|\___|_|_|
                        |___/

Welcome to the Interactive QueryProcessor Shell!
You can access the QueryProcessor instance using the variable 'qp'.
Available tables are stored in 'qp.tables'.
Available table names:
 * Olympics_Country
 * Olympics_Games
 * Olympic_Athlete_Bio
 * Olympic_Athlete_Event_Results
 * Olympic_Games_Medal_Tally
 * Olympic_Results

In [1]: games_city_proj = qp.project(table="Olympics_Games", columns=['city'])

In [2]: prov = qp.select(table=games_city_proj, condition="city == 'Paris'").iloc[0]['provenance']

In [3]: prov
Out[3]: X_Olympics_Games_32 + (X_Olympics_Games_1 + X_Olympics_Games_7)

In [4]: for atom in prov.atoms():
   ...:     print(f"{atom=}, prob={qp.provenance_probability_map[atom]}")
   ...:
atom=X_Olympics_Games_32, prob=0.02288599276242753
atom=X_Olympics_Games_7, prob=0.8502023835734321
atom=X_Olympics_Games_1, prob=0.2676842826507083

In [5]: exact_prob = qp.compute_probability(provenance_expr=prov)

In [6]: monte_carlo_prob = qp.compute_probability(provenance_expr=prov, exact_computation=False)

In [7]: exact_prob, monte_carlo_prob
Out[7]: (0.892811424997835, 0.8951)
```
