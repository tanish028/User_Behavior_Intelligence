import pandas as pd
from data_loader import load_data



def clean_data(df):
    """
    Cleans the raw retail DataFrame by removing nulls,
    fixing data types, and filtering invalid transactions.

    Args:
        df (pd.DataFrame): Raw DataFrame from load_data()

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis
    """

    df = df.drop_duplicates()
    df = df.rename(columns={"Customer ID": "CustomerID"})
    df = df.dropna(subset = ["CustomerID","Description"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)


    df = df[df["Quantity"]>0]
    df = df[df["Price"] > 0]

    # Cap extreme quantities - orders above 10,000 are likely data errors
    df = df[df["Quantity"] < 10000]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    df = df.reset_index(drop = True)
    return df


if __name__ == "__main__":
    df = load_data("data/online_retail_II.xlsx")
    print(df.shape)
    cleaned_data = clean_data(df)
    print(cleaned_data.shape)
    cleaned_data.to_csv("data/cleaned_data.csv", index=False)
    print("Cleaned data saved to data/cleaned_data.csv")



