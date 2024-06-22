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

double implied_volatility(double S, double K, double T, double r, int N, const std::string& option_type, double market_price, double tol = 1e-6, int max_iter = 1000) {
    auto f = [&](double sigma) {
        return binomial_tree_american_option(S, K, T, r, sigma, N, option_type) - market_price;
    };

    double a = 0.0001, b = 1000000.0, c, d, e, min1, min2;
    double fa = f(a), fb = f(b), fc, p, q, t, s, tol1, xm;  // Changed 'r' to 't'

    if (fa * fb > 0) {
            std::cout << "failed to converge with inputs: " << S << ", " << K << ", " << T << ", " << r << ", " << N << ", " << option_type << ", " << market_price << "iv: " << b << std::endl;
            exit(1);
        return std::numeric_limits<double>::quiet_NaN();
    }

    c = b;
    fc = fb;

    for (int iter = 0; iter < max_iter; iter++) {
        if (fb * fc > 0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }

        if (std::abs(fc) < std::abs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        tol1 = 2 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        xm = 0.5 * (c - b);

        if (std::abs(xm) <= tol1 || fb == 0) {
            return b;
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            s = fb / fa;
            if (a == c) {
                p = 2 * xm * s;
                q = 1 - s;
            } else {
                q = fa / fc;
                t = fb / fc;  // Using 't' instead of 'r'
                p = s * (2 * xm * q * (q - t) - (b - a) * (t - 1));
                q = (q - 1) * (t - 1) * (s - 1);
            }

            if (p > 0) {
                q = -q;
            }

            p = std::abs(p);
            min1 = 3 * xm * q - std::abs(tol1 * q);
            min2 = std::abs(e * q);

            if (2 * p < (min1 < min2 ? min1 : min2)) {
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

        if (std::abs(d) > tol1) {
            b += d;
        } else {
            b += (xm > 0 ? tol1 : -tol1);
        }

        fb = f(b);
    }
    std::cout << "failed to converge with inputs: " << S << ", " << K << ", " << T << ", " << r << ", " << N << ", " << option_type << ", " << market_price << "iv: " << b << std::endl;
    exit(1);
    return std::numeric_limits<double>::quiet_NaN();
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
                      << std::endl;
        }
    });

    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < options.size(); i += BATCH_SIZE) {
        size_t end = std::min(i + BATCH_SIZE, options.size());
        futures.push_back(pool.enqueue([&, i, end]() {
            size_t local_nan_count = 0;
            for (size_t j = i; j < end; ++j) {
                const auto& option = options[j];
                double iv = implied_volatility(
                    option.underlying_price, option.strike_price, option.years_to_expiration,
                    r, N, option.option_type, option.market_price
                );
                implied_vols[j] = iv;
                if (std::isnan(iv)) {
                    local_nan_count++;
                }
                completed_calculations.fetch_add(1, std::memory_order_relaxed);
            }
            nan_count.fetch_add(local_nan_count, std::memory_order_relaxed);
            
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Batch completed: " << i << " to " << end - 1 
                      << " | NaN count in this batch: " << local_nan_count 
                      << " | Total NaN count: " << nan_count.load() << std::endl;
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

    std::cout << "\nCalculations completed." << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << seconds << " seconds" << std::endl;
    std::cout << "Number of calculations: " << total_calculations << std::endl;
    std::cout << "Overall calculations per second: " << std::setprecision(2) << overall_calculations_per_second << std::endl;
    std::cout << "Total NaN count: " << nan_count.load() << std::endl;

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