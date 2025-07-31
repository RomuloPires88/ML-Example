import pandas as pd
import numpy as np
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def sql():
    """
    Opening a connection to the database
    """
        conn = pyodbc.connect(
        f'DRIVER={{ODBC Driver}};SYSTEM=IP;UID=user;PWD=pass'
    )
    # Testing the database connection
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if result:
            print("Connection established successfully!")
        else:
            print("Unable to verify connection.")
    except pyodbc.Error as e:
        print("Erro to establish a connection:", e)

    # SQL script
    sql = f"""
        SELECT 
            T1.ID, T2.COD, T2.DESCR, T1.DATA, T1.QTD,T3.MELT_POINT, T3.SOLUBILITY, T3.VISCOSITY,
            CASE WHEN T1.Posting = 11 THEN 'raw_material' ELSE 'product' END TYPE
        FROM dbo.Production T1 
        INNER JOIN dbo.CodeBib T2 ON T1.RNKEY = T2.RNKEY
        INNER JOIN lab.QualityTesting T3 ON RIGHT(DIGITS(T1.ID), 8) = T3.ProductionID
        WHERE T1.ProductionLine IN (20, 21, 22)
        AND T1.Posting  IN (11,22)
        """
    # Run the Query
    cursor = conn.cursor()
    cursor.execute(sql)
    # Get columns
    columns = [desc[0] for desc in cursor.description]
    # Search for data
    rows = cursor.fetchall()
    # Convert each row to explicit tuple
    rows_tuples = [tuple(row) for row in rows]
    
    df_production = pd.DataFrame(rows_tuples, columns=columns)
    return df_production

# Run the SQL function and load the results
df_sql = sql()
# Filter and store data in separate DataFrames
df_product = df_sql[df_sql['TYPE']=='product']
df_raw_material = df_sql[df_sql['TYPE']=='raw_material']

# Prepare the matriz X to the correct format
X = df_mp.pivot_table(
    index='ID',
    columns='COD',
    values='QTD'
).infer_objects(copy=False).fillna(0)

# Prepare the target vector y with unique samples indexed by ID
y = df_espumas[['ID', 'MELT_POINT', 'SOLUBILITY', 'VISCOSITY']].drop_duplicates()
y = y.set_index('ID')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Choose the base model for multi-output regression
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)


print(f'\n🔎 Properties Evaluation \n')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot predicted vs. actual values for each property
for i, col in enumerate(y.columns):
    ax = axes[i]
    
    # Scatter real vs previsto
    ax.scatter(y_test[col], y_pred_df[col], alpha=0.7, edgecolors='k')
    ax.plot([y_test[col].min(), y_test[col].max()],
            [y_test[col].min(), y_test[col].max()],
            'r--', lw=2)
    
    # Métricas
    mse = mean_squared_error(y_test[col], y_pred_df[col])
    r2 = r2_score(y_test[col], y_pred_df[col])
    
    # Título com métricas
    ax.set_title(f'{col}\nMSE: {mse:.2f} | R²: {r2:.2f}')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.grid(True)

plt.tight_layout()
plt.show()
