#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

double binomial_tree_american_option(double S, double K, double T, double r, double sigma, int N, const std::string& option_type) {
    double dt = T / N;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1 / u;
    double p = (std::exp(r * dt) - d) / (u - d);
    
    std::vector<std::vector<double>> option_values(N + 1, std::vector<double>(N + 1, 0.0));

    // Initialize option values at maturity
    for (int i = 0; i <= N; ++i) {
        double ST = S * std::pow(u, i) * std::pow(d, N - i);
        if (option_type == "call") {
            option_values[i][N] = std::max(ST - K, 0.0);
        } else if (option_type == "put") {
            option_values[i][N] = std::max(K - ST, 0.0);
        }
    }
    
    // Backward induction
    for (int j = N - 1; j >= 0; --j) {
        for (int i = 0; i <= j; ++i) {
            option_values[i][j] = std::exp(-r * dt) * (p * option_values[i + 1][j + 1] + (1 - p) * option_values[i][j + 1]);
            
            // Early exercise
            double stock_price = S * std::pow(u, i) * std::pow(d, j - i);
            if (option_type == "call") {
                option_values[i][j] = std::max(option_values[i][j], stock_price - K);
            } else if (option_type == "put") {
                option_values[i][j] = std::max(option_values[i][j], K - stock_price);
            }
        }
    }
    
    return option_values[0][0];
}

double implied_volatility(double S, double K, double T, double r, int N, const std::string& option_type, double market_price, double tol = 1e-6, int max_iter = 100) {
    double sigma_low = 0.001, sigma_high = 100;
    for (int i = 0; i < max_iter; ++i) {
        double sigma = (sigma_low + sigma_high) / 2;
        double option_price = binomial_tree_american_option(S, K, T, r, sigma, N, option_type);
        
        if (std::abs(option_price - market_price) < tol) {
            return sigma;
        } else if (option_price < market_price) {
            sigma_low = sigma;
        } else {
            sigma_high = sigma;
        }
    }
    
    return (sigma_low + sigma_high) / 2;
}

struct OptionData {
    double market_price;
    double strike_price;
    double underlying_price;
    double years_to_expiration;
    std::string option_type;
};

std::vector<OptionData> read_csv(const std::string& filename) {
    std::vector<OptionData> options;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return options;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        OptionData option;
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() == 5) {
            try {
                option.market_price = std::stod(tokens[0]);
                option.strike_price = std::stod(tokens[1]);
                option.underlying_price = std::stod(tokens[2]);
                option.years_to_expiration = std::stod(tokens[3]);
                char contract_type = tokens[4][0];
                option.option_type = (contract_type == 'C' || contract_type == 'c') ? "call" : "put";
                options.push_back(option);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Error parsing line: " << line << " - Incorrect number of fields" << std::endl;
        }
    }

    file.close();
    return options;
}

void calculate_implied_volatilities(const std::string& filename, double r, int N) {
    std::vector<OptionData> options = read_csv(filename);
    
    std::cout << "Implied Volatilities:" << std::endl;
    for (const auto& option : options) {
        double implied_vol = implied_volatility(
            option.underlying_price, option.strike_price, option.years_to_expiration,
            r, N, option.option_type, option.market_price
        );
        
        std::cout << "Contract: " << option.option_type 
                  << ", Strike: " << option.strike_price
                  << ", Underlying: " << option.underlying_price
                  << ", Time to Expiry: " << option.years_to_expiration
                  << ", Market Price: " << option.market_price
                  << ", Implied Volatility: " << implied_vol << std::endl;
    }
}

void write_to_csv(const std::string& filename, const std::vector<OptionData>& options, const std::vector<double>& implied_vols) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    // Write header
    outfile << "Contract,Strike,Underlying,TimeToExpiry,MarketPrice,ImpliedVolatility\n";

    // Write data
    for (size_t i = 0; i < options.size(); ++i) {
        const auto& option = options[i];
        outfile << option.option_type << ","
                << option.strike_price << ","
                << option.underlying_price << ","
                << option.years_to_expiration << ","
                << option.market_price << ","
                << std::fixed << std::setprecision(6) << implied_vols[i] << "\n";
    }

    outfile.close();
    std::cout << "Results written to " << filename << std::endl;
}


void calculate_implied_volatilities(const std::string& input_filename, const std::string& output_filename, double r, int N) {
    std::vector<OptionData> options = read_csv(input_filename);
    std::vector<double> implied_vols;
    
    std::cout << "Calculating Implied Volatilities..." << std::endl;
    for (const auto& option : options) {
        double implied_vol = implied_volatility(
            option.underlying_price, option.strike_price, option.years_to_expiration,
            r, N, option.option_type, option.market_price
        );
        implied_vols.push_back(implied_vol);
        
        std::cout << "Contract: " << option.option_type 
                  << ", Strike: " << option.strike_price
                  << ", Underlying: " << option.underlying_price
                  << ", Time to Expiry: " << option.years_to_expiration
                  << ", Market Price: " << option.market_price
                  << ", Implied Volatility: " << implied_vol << std::endl;
    }

    write_to_csv(output_filename, options, implied_vols);
}

int main() {
    std::string input_filename = "/home/qhawkins/Desktop/GMEStudy/timed_opra_clean.csv";
    std::string output_filename = "/home/qhawkins/Desktop/GMEStudy/implied_volatilities.csv";
    double r = 0.0425;    // Risk-free interest rate
    int N = 100;        // Number of time steps

    calculate_implied_volatilities(input_filename, output_filename, r, N);

    return 0;
}