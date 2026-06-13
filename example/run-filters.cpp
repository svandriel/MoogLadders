// Processes input audio through all filter models and writes output WAV files
// Usage: RunFilters -f input.wav -c 1000 -r 0.5

#include "helpers.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cmath>

void PrintHelp(const char* programName) {
    std::cout << "Usage: " << programName << " -f <input.wav> [options]\n\n";
    std::cout << "Processes audio through all filter models and writes output WAV files.\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -f, --file <path>       Input WAV file (required)\n";
    std::cout << "  -c, --cutoff <hz>       Cutoff frequency in Hz (default: 1000.0)\n";
    std::cout << "  -r, --resonance <value> Resonance 0.0-1.0 (default: 0.5)\n";
    std::cout << "  -s, --oversample <n>    Oversampling factor: 0, 2, 4, or 8 (default: 0)\n";
    std::cout << "  -o, --output-dir <dir>  Output directory (default: current directory)\n";
    std::cout << "  --bench                 CPU benchmark mode (no input file needed):\n";
    std::cout << "                          processes an in-memory test signal through every\n";
    std::cout << "                          filter and reports ns/sample as JSON\n";
    std::cout << "  -n, --samples <n>       Benchmark signal length in samples (default: 2097152)\n";
    std::cout << "  -q, --quality <0|1|2>   Hyperion solver quality tier: 0=static, 1=relinearized\n";
    std::cout << "                          (default), 2=outer2. Ignored by other filters.\n";
    std::cout << "\n";
    std::cout << "Output files are named: <FilterName>_c<cutoff>_r<resonance>[_os<factor>].wav\n";
    std::cout << "\n";
    std::cout << "Filter Models:\n";
    for (int i = 0; i < static_cast<int>(FilterModel::Count); ++i) {
        std::cout << "  " << i << " - " << FilterModelNames[i] << "\n";
    }
}

std::string GetBasename(const std::string& path) {
    // Find last path separator
    size_t lastSep = path.find_last_of("/\\");
    std::string filename = (lastSep == std::string::npos) ? path : path.substr(lastSep + 1);

    // Remove extension
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != std::string::npos) {
        filename = filename.substr(0, lastDot);
    }
    return filename;
}

std::string BuildOutputFilename(
    const std::string& outputDir,
    const std::string& basename,
    const char* filterName,
    float cutoff,
    float resonance,
    int oversampleFactor
) {
    char buffer[512];
    if (oversampleFactor > 0) {
        snprintf(buffer, sizeof(buffer), "%s%s%s_c%.0f_r%.2f_os%dx.wav",
            outputDir.c_str(),
            outputDir.empty() ? "" : "/",
            filterName,
            cutoff,
            resonance,
            oversampleFactor);
    } else {
        snprintf(buffer, sizeof(buffer), "%s%s%s_c%.0f_r%.2f.wav",
            outputDir.c_str(),
            outputDir.empty() ? "" : "/",
            filterName,
            cutoff,
            resonance);
    }
    return std::string(buffer);
}

// Hyperion overrides (applied only to the non-oversampled Hyperion variants):
// solver quality tier (-1 = leave at filter default) and ADAA enable.
static int hyperionQuality = -1;
static bool hyperionAdaa = true;

static void ApplyHyperionQuality(LadderFilterBase* f)
{
    if (auto* h = dynamic_cast<HyperionMoog*>(f)) {
        if (hyperionQuality >= 0) h->SetQuality(static_cast<HyperionMoog::SolverQuality>(hyperionQuality));
        h->SetAdaaEnabled(hyperionAdaa);
    } else if (auto* ht = dynamic_cast<HyperionMoogTanh*>(f)) {
        if (hyperionQuality >= 0) ht->SetQuality(static_cast<HyperionMoogTanh::SolverQuality>(hyperionQuality));
        ht->SetAdaaEnabled(hyperionAdaa);
    }
}

// Deterministic benchmark signal: 110 Hz saw + white noise (LCG), both at 0.25.
// Deterministic so the printed checksum is reproducible across runs/builds.
static std::vector<float> GenerateBenchSignal(int n, int sampleRate)
{
    std::vector<float> s(static_cast<size_t>(n));
    uint32_t rng = 0x12345678u;
    double phase = 0.0;
    const double inc = 110.0 / sampleRate;
    for (int i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        const float noise = ((rng >> 8) * (1.0f / 8388608.0f)) - 1.0f;
        const float saw = static_cast<float>(2.0 * phase - 1.0);
        phase += inc;
        if (phase >= 1.0) phase -= 1.0;
        s[i] = 0.25f * saw + 0.25f * noise;
    }
    return s;
}

static int RunBenchmark(int numSamples, float cutoff, float resonance, const std::string& outputDir)
{
    const int sampleRate = 44100;
    const int reps = 5;
    const int warmupSamples = 10000;

    const std::vector<float> input = GenerateBenchSignal(numSamples, sampleRate);

    std::cout << "Benchmark: " << numSamples << " samples @ " << sampleRate << " Hz, "
              << reps << " reps (min), cutoff=" << cutoff << " Hz, resonance=" << resonance << "\n";
    std::cout << "=========================================\n";

    std::ostringstream json;
    json << "[\n";

    const int filterCount = static_cast<int>(FilterModel::Count);
    for (int i = 0; i < filterCount; ++i) {
        const FilterModel model = static_cast<FilterModel>(i);
        const char* filterName = FilterModelNames[i];

        auto filter = CreateFilter(model, static_cast<float>(sampleRate));
        ApplyHyperionQuality(filter.get());
        filter->SetCutoff(cutoff);
        filter->SetResonance(resonance);

        // Warmup: settle filter state and caches
        std::vector<float> warm(input.begin(), input.begin() + std::min(numSamples, warmupSamples));
        filter->Process(warm.data(), static_cast<uint32_t>(warm.size()));

        double bestNsPerSample = 1e300;
        double checksum = 0.0;
        for (int r = 0; r < reps; ++r) {
            std::vector<float> buf = input;
            const auto t0 = std::chrono::steady_clock::now();
            filter->Process(buf.data(), static_cast<uint32_t>(buf.size()));
            const auto t1 = std::chrono::steady_clock::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / numSamples;
            bestNsPerSample = std::min(bestNsPerSample, ns);
            // Checksum defeats dead-code elimination and doubles as a sanity check
            checksum = 0.0;
            for (const float v : buf) checksum += v;
        }

        const bool finite = std::isfinite(checksum);
        std::cout << filterName << ": " << bestNsPerSample << " ns/sample"
                  << (finite ? "" : "  [WARNING: non-finite output]") << "\n";

        json << "  {\"filter\": \"" << filterName << "\""
             << ", \"ns_per_sample\": " << bestNsPerSample
             << ", \"samples\": " << numSamples
             << ", \"reps\": " << reps
             << ", \"cutoff\": " << cutoff
             << ", \"resonance\": " << resonance
             << ", \"checksum\": " << checksum << "}"
             << (i + 1 < filterCount ? "," : "") << "\n";
    }
    json << "]\n";

    std::cout << "\n" << json.str();

    if (!outputDir.empty()) {
        const std::string path = outputDir + "/bench.json";
        std::ofstream f(path);
        if (f) {
            f << json.str();
            std::cout << "Wrote " << path << "\n";
        } else {
            std::cerr << "Failed to write " << path << "\n";
            return 1;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    std::string inputFile;
    std::string outputDir;
    float cutoff = 1000.0f;
    float resonance = 0.5f;
    int oversampleFactor = 0;
    bool benchMode = false;
    int benchSamples = 1 << 21;
    hyperionQuality = -1; // -1 = leave at filter default

    // Parse cli args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            PrintHelp(argv[0]);
            return 0;
        }
        else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            inputFile = argv[++i];
        }
        else if ((arg == "-c" || arg == "--cutoff") && i + 1 < argc) {
            cutoff = static_cast<float>(std::atof(argv[++i]));
            if (cutoff <= 0.0f) {
                std::cerr << "Cutoff (hz) must be positive.\n";
                return 1;
            }
        }
        else if ((arg == "-r" || arg == "--resonance") && i + 1 < argc) {
            resonance = static_cast<float>(std::atof(argv[++i]));
            if (resonance < 0.0f || resonance > 1.0f) {
                std::cerr << "Resonance should be between 0.0 and 1.0.\n";
                return 1;
            }
        }
        else if ((arg == "-s" || arg == "--oversample") && i + 1 < argc) {
            oversampleFactor = std::atoi(argv[++i]);
            if (oversampleFactor != 0 && oversampleFactor != 2 && oversampleFactor != 4 && oversampleFactor != 8) {
                std::cerr << "Oversampling factor must be 0, 2, 4, or 8.\n";
                return 1;
            }
        }
        else if ((arg == "-o" || arg == "--output-dir") && i + 1 < argc) {
            outputDir = argv[++i];
        }
        else if (arg == "--bench") {
            benchMode = true;
        }
        else if ((arg == "-n" || arg == "--samples") && i + 1 < argc) {
            benchSamples = std::atoi(argv[++i]);
            if (benchSamples < 1024) {
                std::cerr << "Benchmark sample count must be >= 1024.\n";
                return 1;
            }
        }
        else if (arg == "--no-adaa") {
            hyperionAdaa = false;
        }
        else if ((arg == "-q" || arg == "--quality") && i + 1 < argc) {
            hyperionQuality = std::atoi(argv[++i]);
            if (hyperionQuality < 0 || hyperionQuality > 2) {
                std::cerr << "Quality tier must be 0, 1, or 2.\n";
                return 1;
            }
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
    }

    if (benchMode) {
        return RunBenchmark(benchSamples, cutoff, resonance, outputDir);
    }

    if (inputFile.empty()) {
        std::cerr << "Error: Input file is required. Use -f <input.wav>\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    // Load input WAV file
    int sampleRate = 0;
    int numChannels = 0;
    std::vector<float> inputSamples;

    std::cout << "Loading: " << inputFile << "\n";
    if (!ReadWavFile(inputFile.c_str(), sampleRate, numChannels, inputSamples)) {
        std::cerr << "Failed to load WAV file: " << inputFile << "\n";
        return 1;
    }
    std::cout << "Loaded " << inputSamples.size() << " samples, "
              << sampleRate << " Hz, " << numChannels << " channels\n";

    std::string basename = GetBasename(inputFile);

    std::cout << "\nProcessing with cutoff=" << cutoff << " Hz, resonance=" << resonance;
    if (oversampleFactor > 0) {
        std::cout << ", oversampling=" << oversampleFactor << "x";
    }
    std::cout << "\n";
    std::cout << "=========================================\n\n";

    int successCount = 0;
    int filterCount = static_cast<int>(FilterModel::Count);

    for (int i = 0; i < filterCount; ++i) {
        FilterModel model = static_cast<FilterModel>(i);
        const char* filterName = FilterModelNames[i];

        std::cout << "Processing with " << filterName;
        if (oversampleFactor > 0) {
            std::cout << " (" << oversampleFactor << "x)";
        }
        std::cout << "... ";
        std::cout.flush();

        // Create a copy of input samples for this filter
        std::vector<float> samples = inputSamples;

        // Create and configure filter (with or without oversampling)
        std::unique_ptr<LadderFilterBase> filter;
        if (oversampleFactor > 0) {
            MoogLadders::OversamplingPreset preset;
            switch (oversampleFactor) {
                case 2: preset = MoogLadders::OversamplingPreset::X2; break;
                case 4: preset = MoogLadders::OversamplingPreset::X4; break;
                case 8: preset = MoogLadders::OversamplingPreset::X8; break;
                default: preset = MoogLadders::OversamplingPreset::X2; break;
            }
            filter = CreateOversampledFilter(model, static_cast<float>(sampleRate), preset);
        } else {
            filter = CreateFilter(model, static_cast<float>(sampleRate));
        }
        ApplyHyperionQuality(filter.get());
        filter->SetCutoff(cutoff);
        filter->SetResonance(resonance);

        // Build output filename
        std::string outputFile = BuildOutputFilename(outputDir, basename, filterName, cutoff, resonance, oversampleFactor);

        // Process
        {
            ScopedTimer t(outputFile);
            filter->Process(samples.data(), static_cast<uint32_t>(samples.size()));
        }


        // Write output
        if (WriteWavFile(outputFile.c_str(), sampleRate, numChannels, samples)) {
            std::cout << "OK -> " << outputFile << "\n";
            successCount++;
        } else {
            std::cout << "FAILED to write " << outputFile << "\n";
        }
    }

    std::cout << "\n=========================================\n";
    std::cout << "Processed " << successCount << "/" << filterCount << " filters successfully.\n";

    return (successCount == filterCount) ? 0 : 1;
}
