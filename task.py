import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import random
from tqdm import tqdm

def read_data_from_db(file_path):
    conn = sqlite3.connect(file_path)
    
    try:
        cursor = conn.cursor()
        
        # Retrieve the first table name from the sqlite_master table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        result = cursor.fetchone()
        
        if result is None:
            print("No tables found in the database.")
            conn.close()
            return None
        
        table_name = result[0]
        
        # Read the data from the retrieved table
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data
    
    except sqlite3.Error as e:
        print(f"An error occurred while reading the database: {e}")
        return None

def load_data(path):
    try:
        _, file_extension = os.path.splitext(path)

        if file_extension == ".csv":
            data = pd.read_csv(path)
        elif file_extension in [".xls", ".xlsx"]:
            data = pd.read_excel(path)
        elif file_extension in [".db",".sql"]:
            data = read_data_from_db(path)

        else:
            raise ValueError(
                "Unsupported file format. Please provide a CSV, Excel, SQL, or SQLite database file."
            )

        return data

    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None

    except pd.errors.ParserError:
        print(
            "Error occurred while parsing the file. Please check if the file format is correct."
        )
        return None

    except Exception as e:
        print("An error occurred while loading the data:", str(e))
        return None


def preprocess_data(data):
    # Identify column types
    categorical_columns = data.select_dtypes(include="object").columns
    numerical_columns = data.select_dtypes(include=["int", "float"]).columns

    # Handle missing values in numerical columns
    data[numerical_columns] = data[numerical_columns].fillna(
        data[numerical_columns].mean()
    )

    # Handle missing values in categorical columns
    data[categorical_columns] = data[categorical_columns].fillna(
        data[categorical_columns].mode().iloc[0]
    )

    # Encode categorical features
    encoded_data = pd.get_dummies(data, columns=categorical_columns)

    # Scale numerical features
    scaled_data = (
        encoded_data[numerical_columns] - encoded_data[numerical_columns].mean()
    ) / encoded_data[numerical_columns].std()
    encoded_data[numerical_columns] = scaled_data

    # Rename columns with camel case
    encoded_data.columns = encoded_data.columns.str.replace("_", "")

    return encoded_data


def visualize_individual_column(data, header_name):
    plt.figure(figsize=(12, 8))

    if header_name in data.columns:
        column_data = data[header_name]

        if column_data.dtype == "object":
            # Categorical column visualization
            value_counts = column_data.value_counts()

            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"{header_name} Histogram")
            plt.xlabel(header_name)
            plt.ylabel("Count")

            # Display value counts on the plot
            for i, count in enumerate(value_counts.values):
                plt.text(i, count, str(count), ha="center", va="bottom")

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            print(f"\nValue counts for column '{header_name}':")
            print(value_counts)
        else:
            # Numerical column visualization
            sns.histplot(
                column_data, kde=True, color=random.choice(list(sns.color_palette()))
            )
            plt.title(f"{header_name} Histogram")
            plt.xlabel(header_name)
            plt.ylabel("Frequency")

            # Display descriptive statistics
            descriptive_stats = column_data.describe()
            print(f"\nDescriptive statistics for column '{header_name}':")
            print(descriptive_stats)
            plt.tight_layout()
            plt.show()
    else:
        print(f"Column '{header_name}' does not exist in the data.")



def visualize_boxplot(data, x_column, y_column):
    visualization_name = "Boxplot"

    # Display progress bar
    with tqdm(total=len(data)) as pbar:
        plt.figure(figsize=(8, 6))

        if data[x_column].dtype == "object":
            # Categorical x column and numerical y column
            unique_x = data[x_column].unique()
            colors = random.choices(list(sns.color_palette()), k=len(unique_x))

            sns.boxplot(data=data, x=x_column, y=y_column, palette=colors)
            plt.title(f"{x_column} vs {y_column} {visualization_name}")
            plt.xlabel(x_column)
            plt.ylabel(y_column)

        elif data[y_column].dtype == "object":
            # Numerical x column and categorical y column
            unique_y = data[y_column].unique()
            colors = random.choices(list(sns.color_palette()), k=len(unique_y))

            sns.boxplot(data=data, x=x_column, y=y_column, palette=colors)
            plt.title(f"{x_column} vs {y_column} {visualization_name}")
            plt.xlabel(x_column)
            plt.ylabel(y_column)

        else:
            # Numerical x and y columns
            sns.boxplot(data=data, x=x_column, y=y_column)
            plt.title(f"{x_column} vs {y_column} {visualization_name}")
            plt.xlabel(x_column)
            plt.ylabel(y_column)

        plt.tight_layout()
        plt.show()

        pbar.update(len(data))
        
def visualize_scatterplot(data, x_column, y_column):
    visualization_type = "Scatterplot"

    # Display progress bar
    with tqdm(total=len(data)) as pbar:
        plt.figure(figsize=(8, 6))

        if data[x_column].dtype == "object" and data[y_column].dtype == "object":
            # Categorical x and y columns
            unique_x = data[x_column].unique()
            colors = random.choices(list(sns.color_palette()), k=len(unique_x))

            sns.scatterplot(data=data, x=x_column, y=y_column, hue=x_column, palette=colors)

        elif data[x_column].dtype == "object":
            # Categorical x column and numerical y column
            unique_x = data[x_column].unique()
            colors = random.choices(list(sns.color_palette()), k=len(unique_x))

            sns.stripplot(data=data, x=x_column, y=y_column, hue=x_column, palette=colors)

        elif data[y_column].dtype == "object":
            # Numerical x column and categorical y column
            unique_y = data[y_column].unique()
            colors = random.choices(list(sns.color_palette()), k=len(unique_y))

            sns.stripplot(data=data, x=x_column, y=y_column, hue=y_column, palette=colors)

        else:
            # Numerical x and y columns
            sns.scatterplot(data=data, x=x_column, y=y_column)

        plt.title(f"{x_column} vs {y_column} {visualization_type}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout()
        plt.show()

        pbar.update(len(data))

def visualize_pie_chart(data, column):
    visualization_type = "Pie Chart"

    # Count the occurrences of each unique value in the column
    value_counts = data[column].value_counts()

    # Get the labels and corresponding counts
    labels = value_counts.index
    counts = value_counts.values

    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))

    plt.title(f"{column} {visualization_type}")
    plt.axis('equal')
    plt.show()

        
def main():
    try:
        file_path = input("Enter the path to the data file: ")
        data = load_data(file_path)
        print("\nDataFrame:")
        print(data)
        preprocessed_data = preprocess_data(data)

        while True:
            print("\nVisualization Options:")
            print("1. Visualize Histogram ")
            print("2. Visualize Box Plot ")
            print("3. Visualize Scatter Plot ")
            print("4. Visualize Pie Chart ")
            print("5. Exit ")

            choice = int(
                input(
                    "Enter the number corresponding to the visualization option you want to choose: "
                )
            )

            if choice == 5:
                break
            elif choice == 1:
                try:
                        type = int(input("\nChoose type:\n1. header, 2. column values: "))
                        if type == 2:
                            data = preprocessed_data
                        elif type != 1:
                            raise ValueError
                except ValueError:
                        print("Invalid input. Please enter either 1 or 2.")
                        choice = input("Enter 'exit' to quit or press any key to try again: ")
                        if choice.lower() == 'exit':
                            exit()
                print("\nAvailable Columns:")
                for i, column in enumerate(data.columns):
                    print(f"{i}. {column}")

                column_choice = int(
                    input(
                        "Enter the number corresponding to the column you want to visualize: "
                    )
                )
                if 0 <= column_choice < len(data.columns):
                    column = data.columns[column_choice]
                    visualize_individual_column(data, column)
                else:
                    print("Invalid choice. Please try again.")
            elif choice == 2:
                while True:
                    try:
                        type = int(input("\nChoose comparison type:\n1. Compare between two headers, 2. Compare between two column values: "))
                        if type == 2:
                            data = preprocessed_data
                        elif type != 1:
                            raise ValueError
                    except ValueError:
                        print("Invalid input. Please enter either 1 or 2.")
                        choice = input("Enter 'exit' to quit or press any key to try again: ")
                        if choice.lower() == 'exit':
                            exit()
                    print("\nAvailable Columns:")
                    for i, column in enumerate(data.columns):
                        print(f"{i}. {column}")

                    x_column_choice = int(
                        input("Enter the number corresponding to the x-axis column: ")
                    )
                    y_column_choice = int(
                        input("Enter the number corresponding to the y-axis column: ")
                    )
                    if 0 <= x_column_choice < len(
                        data.columns
                    ) and 0 <= y_column_choice < len(data.columns):
                        x_column = data.columns[x_column_choice]
                        y_column = data.columns[y_column_choice]
                        visualize_boxplot(data, x_column, y_column)
                    else:
                        print("Invalid choice. Please try again.")
            elif choice == 3:
                try:
                    type = int(input("\nChoose comparison type:\n1. Compare between two headers, 2. Compare between two column values: "))
                    if type == 2:
                        data = preprocessed_data
                    elif type != 1:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Please enter either 1 or 2.\n")
                print("\nAvailable Columns:")
                for i, column in enumerate(data.columns):
                    print(f"{i}. {column}")

                x_column_choice = int(
                    input("Enter the number corresponding to the x-axis column: ")
                )
                y_column_choice = int(
                    input("Enter the number corresponding to the y-axis column: ")
                )
                if 0 <= x_column_choice < len(
                    data.columns
                ) and 0 <= y_column_choice < len(data.columns):
                    x_column = data.columns[x_column_choice]
                    y_column = data.columns[y_column_choice]
                    visualize_scatterplot(data, x_column, y_column)
                else:
                    print("Invalid choice. Please try again.")
            elif choice == 4:
                try:
                        type = int(input("\nChoose type:\n1. header, 2. column values: "))
                        if type == 2:
                            data = preprocessed_data
                        elif type != 1:
                            raise ValueError
                except ValueError:
                        print("Invalid input. Please enter either 1 or 2.")
                        choice = input("Enter 'exit' to quit or press any key to try again: ")
                        if choice.lower() == 'exit':
                            exit()
                print("\nAvailable Columns:")
                for i, column in enumerate(data.columns):
                    print(f"{i}. {column}")

                column_choice = int(input("Enter the number corresponding to the column you want to visualize: "))
                if 0 <= column_choice < len(data.columns):
                    column = data.columns[column_choice]
                    visualize_pie_chart(data, column)
                else:
                    print("Invalid choice. Please try again.")
            else:
                print("Invalid choice. Please try again.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
