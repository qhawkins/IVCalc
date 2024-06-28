# Implied Volatility Calculator

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Code Structure](#code-structure)
7. [Key Components](#key-components)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)

## Introduction

The Implied Volatility Calculator is a high-performance C++ program designed to calculate implied volatilities for a large number of options contracts efficiently. It uses a multi-threaded approach to process option data in batches, making it suitable for handling extensive datasets.

The program implements the binomial tree model for American options pricing and uses Brent's method for root-finding to calculate implied volatilities. It's designed to handle various edge cases and provide detailed progress information during the calculation process.

## Features

- Fast, multi-threaded calculation of implied volatilities
- Support for both call and put American options
- Efficient binomial tree model implementation
- Robust implied volatility calculation using Brent's method
- CSV input and output for easy integration with other tools
- Detailed progress reporting and error handling
- Automatic handling of edge cases

## Requirements

- C++11 or later
- CMake 3.10 or later (for building)
- A C++ compiler supporting C++11 features (e.g., GCC, Clang, MSVC)

## Usage

1. Prepare your input CSV file with the following columns:
   - Market Price
   - Strike Price
   - Underlying Price
   - Years to Expiration
   - Option Type (C for Call, P for Put)

2. Update the `main()` function in the source code with your input and output file paths, risk-free rate, and number of time steps:

   ```cpp
   std::string input_filename = "path/to/your/input.csv";
   std::string output_filename = "path/to/your/output.csv";
   double r = 0.0425;    // Risk-free interest rate
   int N = 100;          // Number of time steps
   ```

3. Run the program:
   ```
   ./main
   ```

4. The program will display progress information and write the results to the specified output CSV file.

## Code Structure

The main components of the program are:

- `ThreadPool`: A class for managing concurrent task execution
- `binomial_tree_american_option`: Function to price American options using the binomial tree model
- `implied_volatility`: Function to calculate implied volatility using Brent's method
- `read_csv`: Function to read option data from a CSV file
- `write_to_csv`: Function to write calculated implied volatilities to a CSV file
- `calculate_implied_volatilities`: Main function orchestrating the parallel calculation of implied volatilities
- `main`: Entry point of the program, setting up parameters and calling `calculate_implied_volatilities`

## Key Components

### ThreadPool

The `ThreadPool` class manages a pool of worker threads for efficient task distribution. It allows for concurrent execution of tasks, significantly improving the performance of the implied volatility calculations.

### Binomial Tree Model

The `binomial_tree_american_option` function implements the binomial tree model for pricing American options. This model is particularly suitable for American options as it can handle early exercise scenarios.

### Implied Volatility Calculation

The `implied_volatility` function uses Brent's method for root-finding to calculate the implied volatility. This method is chosen for its reliability and efficiency in finding the root of the pricing function.

### CSV Handling

The `read_csv` and `write_to_csv` functions handle input and output operations, allowing for easy integration with other tools and data sources.

## Performance Considerations

- The program uses multi-threading to parallelize calculations, significantly improving performance on multi-core systems.
- Calculations are performed in batches to optimize memory usage and improve cache efficiency.
- The binomial tree model is implemented with a fixed-size array to avoid dynamic memory allocations during the pricing process.
- Edge cases are handled efficiently to avoid unnecessary calculations.

## Troubleshooting

- If you encounter "NaN" or "Root not bracketed" errors in the output, check your input data for potential issues such as:
  - Negative or zero prices
  - Unrealistic strike prices
  - Expired options (time to expiration <= 0)
- Adjust the `BATCH_SIZE` constant in the `calculate_implied_volatilities` function if you're experiencing memory issues or want to fine-tune performance.
- If the program is too slow, try increasing the number of threads in the `ThreadPool` constructor call.
