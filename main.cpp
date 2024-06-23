#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <chrono>
#include <atomic>
#include <array>
#include <limits>
#include <random>

class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for(size_t i = 0; i < num_threads; ++i)
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using return_type = typename std::result_of<F()>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

constexpr int MAX_N = 100;  // Maximum number of time steps

double binomial_tree_american_option(double S, double K, double T, double r, double sigma, int N, const std::string& option_type) {
    double dt = T / N;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1 / u;
    double p = (std::exp(r * dt) - d) / (u - d);
    
    std::array<std::array<double, MAX_N + 1>, MAX_N + 1> option_values;

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

double implied_volatility(double S, double K, double T, double r, int N, const std::string& option_type, double market_price, double tol = 1e-8, int max_iter = 250) {
    auto f = [&](double sigma) {
        return binomial_tree_american_option(S, K, T, r, sigma, N, option_type) - market_price;
    };

    // Initial guess
    double a = 0.00001;
    double b = 100.0;
    double fa = f(a);
    double fb = f(b);

    // If not bracketed, expand the interval
    int bracket_attempts = 0;
    while (fa * fb > 0 && bracket_attempts < 50) {
        if (std::abs(fa) < std::abs(fb)) {
            a -= (b - a);
            fa = f(a);
        } else {
            b += (b - a);
            fb = f(b);
        }
        bracket_attempts++;
    }

    if (fa * fb > 0) {
        return -1;  // Root not bracketed after attempts
    }

    double c = b, fc = fb;
    double d, e;

    for (int iter = 0; iter < max_iter; iter++) {
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a; fc = fa;
            d = b - a; e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }

        double tol1 = 2 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        double xm = 0.5 * (c - b);
        
        if (std::abs(xm) <= tol1 || fb == 0) {
            return b;  // Found a solution
        }
        
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2 * xm * s;
                q = 1 - s;
            } else {
                q = fa / fc;
                double r = fb / fc;
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1));
                q = (q - 1) * (r - 1) * (s - 1);
            }
            if (p > 0) q = -q;
            p = std::abs(p);
            
            if (2 * p < std::min(3 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;
        b += (std::abs(d) > tol1) ? d : (xm > 0 ? tol1 : -tol1);
        fb = f(b);
    }

    return -2;  // Max iterations reached
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

        if (tokens.size() == 10) {
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

void write_to_csv(const std::string& filename, const std::vector<OptionData>& options, const std::vector<double>& implied_vols) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    outfile << "Contract,Strike,Underlying,TimeToExpiry,MarketPrice,ImpliedVolatility\n";

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
    std::vector<double> implied_vols(options.size(), 0.0);
    
    ThreadPool pool(std::thread::hardware_concurrency() - 1);

    std::cout << "Calculating Implied Volatilities..." << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<size_t> completed_calculations(0);
    std::atomic<bool> calculation_complete(false);
    std::atomic<size_t> nan_count(0);
    std::atomic<size_t> root_not_bracketed_count(0);
    std::atomic<size_t> max_iterations_reached_count(0);
    size_t total_calculations = options.size();
    const size_t BATCH_SIZE = 1000;  // Adjust based on your system

    std::mutex cout_mutex;

    // Start a thread to print progress
    std::thread progress_thread([&]() {
        auto next_print_time = start_time;
        while (!calculation_complete) {
            next_print_time += std::chrono::seconds(1);
            std::this_thread::sleep_until(next_print_time);
            
            auto current_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            size_t current_completed = completed_calculations.load();
            double calculations_per_second = current_completed / (duration.count() / 1000.0);
            
            size_t calculations_left = total_calculations - current_completed;
            double estimated_time_left = calculations_per_second > 0 ? calculations_left / calculations_per_second : 0;
            
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Calculations per second: " << std::fixed << std::setprecision(2) << calculations_per_second
                      << " | Completed: " << current_completed << "/" << total_calculations
                      << " | Left: " << calculations_left
                      << " | Estimated time left: " << std::setprecision(1) << estimated_time_left << " seconds"
                      << " | NaN count: " << nan_count.load()
                      << " | Root not bracketed: " << root_not_bracketed_count.load()
                      << " | Max iterations reached: " << max_iterations_reached_count.load()
                      << std::endl;
        }
    });

    std::vector<std::future<void>> futures;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, options.size() - 1);

    for (size_t i = 0; i < options.size(); i += BATCH_SIZE) {
        size_t end = std::min(i + BATCH_SIZE, options.size());
        futures.push_back(pool.enqueue([&, i, end]() {
            size_t local_nan_count = 0;
            size_t local_root_not_bracketed_count = 0;
            size_t local_max_iterations_reached_count = 0;
            double sum_iv = 0.0;
            size_t valid_iv_count = 0;

            for (size_t j = i; j < end; ++j) {
                const auto& option = options[j];
                
                // Data sanity checks
                if (option.market_price <= 0 || option.strike_price <= 0 || 
                    option.underlying_price <= 0 || option.years_to_expiration <= 0) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cerr << "Invalid option data for index " << j << std::endl;
                    std::cout << "  Market price: " << option.market_price
                              << ", Strike price: " << option.strike_price
                              << ", Underlying price: " << option.underlying_price
                              << ", Years to expiration: " << option.years_to_expiration
                              << ", Option type: " << option.option_type << std::endl;
                    exit(1);
                    local_nan_count++;
                    implied_vols[j] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }

                // Handle edge cases
                if (option.market_price < 0.01) {
                    implied_vols[j] = 0.01;
                    sum_iv += 0.01;
                    valid_iv_count++;
                    continue;
                }
                if (option.option_type == "call" && option.market_price > option.underlying_price) {
                    implied_vols[j] = 5.0;
                    sum_iv += 5.0;
                    valid_iv_count++;
                    continue;
                }

                double iv = implied_volatility(
                    option.underlying_price, option.strike_price, option.years_to_expiration,
                    r, N, option.option_type, option.market_price
                );

                // Print detailed information for a random sample of options
                if (j == dis(gen)) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "Detailed info for option " << j << ":\n"
                              << "  Underlying: " << option.underlying_price
                              << ", Strike: " << option.strike_price
                              << ", Time to expiry: " << option.years_to_expiration
                              << ", Market price: " << option.market_price
                              << ", Option type: " << option.option_type
                              << ", Calculated IV: " << iv << std::endl;
                }

                if (iv == -1) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "Root not bracketed for option " << j << ":\n"
                              << "  Underlying: " << option.underlying_price
                              << ", Strike: " << option.strike_price
                              << ", Time to expiry: " << option.years_to_expiration
                              << ", Market price: " << option.market_price
                              << ", Option type: " << option.option_type << std::endl;
                    local_root_not_bracketed_count++;
                    implied_vols[j] = std::numeric_limits<double>::quiet_NaN();
                } else if (iv == -2) {
                    local_max_iterations_reached_count++;
                    implied_vols[j] = std::numeric_limits<double>::quiet_NaN();
                } else if (std::isnan(iv)) {
                    local_nan_count++;
                    implied_vols[j] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    implied_vols[j] = iv;
                    sum_iv += iv;
                    valid_iv_count++;
                }
                completed_calculations.fetch_add(1, std::memory_order_relaxed);
            }

            nan_count.fetch_add(local_nan_count, std::memory_order_relaxed);
            root_not_bracketed_count.fetch_add(local_root_not_bracketed_count, std::memory_order_relaxed);
            max_iterations_reached_count.fetch_add(local_max_iterations_reached_count, std::memory_order_relaxed);
            
            double avg_iv = valid_iv_count > 0 ? sum_iv / valid_iv_count : 0.0;

            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Batch completed: " << i << " to " << end - 1 
                      << " | NaN count in this batch: " << local_nan_count 
                      << " | Root not bracketed in this batch: " << local_root_not_bracketed_count
                      << " | Max iterations reached in this batch: " << local_max_iterations_reached_count
                      << " | Average IV in this batch: " << std::fixed << std::setprecision(4) << avg_iv
                      << std::endl;
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    calculation_complete = true;
    progress_thread.join();

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double seconds = duration.count() / 1000.0;
    double overall_calculations_per_second = total_calculations / seconds;

    // Calculate overall average IV
    double sum_iv = 0.0;
    size_t valid_iv_count = 0;
    for (double iv : implied_vols) {
        if (!std::isnan(iv)) {
            sum_iv += iv;
            valid_iv_count++;
        }
    }
    double overall_avg_iv = valid_iv_count > 0 ? sum_iv / valid_iv_count : 0.0;

    std::cout << "\nCalculations completed." << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << seconds << " seconds" << std::endl;
    std::cout << "Number of calculations: " << total_calculations << std::endl;
    std::cout << "Overall calculations per second: " << std::setprecision(2) << overall_calculations_per_second << std::endl;
    std::cout << "Total NaN count: " << nan_count.load() << std::endl;
    std::cout << "Total root not bracketed count: " << root_not_bracketed_count.load() << std::endl;
    std::cout << "Total max iterations reached count: " << max_iterations_reached_count.load() << std::endl;
    std::cout << "Overall average IV: " << std::fixed << std::setprecision(4) << overall_avg_iv << std::endl;

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