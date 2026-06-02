import pandas as pd

def load_data(file_path):
    # Reading a specific sheet by name
    df1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

    # Stacking two DataFrames vertically
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data("data/online_retail_II.xlsx")

    print(df.head())


