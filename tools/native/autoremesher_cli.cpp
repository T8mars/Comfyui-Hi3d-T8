// Hi3D workbench console adapter for the separately licensed AutoRemesher core.
// Geometry algorithms remain in the pinned upstream source tree.
#include <AutoRemesher/AutoRemesher>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

using Mesh = AutoRemesher::AutoRemesher;
using Vector = AutoRemesher::Vector3;

static void progress(void*, float fraction, const char* status) {
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    // One buffered write keeps core stderr diagnostics out of progress records.
    std::ostringstream line;
    line << std::fixed << std::setprecision(2) << fraction * 100
         << "% done. " << (status ? status : "") << '\n';
    const auto text = line.str();
    std::cout.write(text.data(), static_cast<std::streamsize>(text.size()));
    std::cout.flush();
}

static int run(const std::vector<std::string>& args) {
    std::map<std::string, std::string> options;
    const std::vector<std::string> allowed = {"--input", "--output", "--report", "--target-quads",
        "--edge-scaling", "--sharp-edge", "--smooth-normal", "--adaptivity", "--anisotropy"};
    for (size_t i = 1; i < args.size(); i += 2) {
        if (i + 1 >= args.size() || std::find(allowed.begin(), allowed.end(), args[i]) == allowed.end()
                || !options.emplace(args[i], args[i + 1]).second)
            throw std::runtime_error("Invalid or duplicate command argument");
    }
    if (!options.count("--input") || !options.count("--output") || !options.count("--target-quads"))
        throw std::runtime_error("Required: --input --output --target-quads");
    const auto number = [&](const char* name, double fallback, double low, double high) {
        auto entry = options.find(name);
        if (entry == options.end()) return fallback;
        size_t end = 0;
        double result = std::stod(entry->second, &end);
        if (end != entry->second.size() || !std::isfinite(result) || result < low || result > high)
            throw std::runtime_error(std::string("Invalid value: ") + name);
        return result;
    };
    double requested = number("--target-quads", 15000, 100, 1000000);
    if (requested != std::floor(requested)) throw std::runtime_error("Target quads must be integral");
    const auto input = std::filesystem::u8path(options.at("--input"));
    const auto output = std::filesystem::u8path(options.at("--output"));
    std::error_code identity_error;
    if (std::filesystem::equivalent(input, output, identity_error))
        throw std::runtime_error("Input and output must differ");
    std::ifstream stream(input);
    if (!stream) throw std::runtime_error("Cannot read input OBJ");
    std::vector<Vector> vertices;
    std::vector<std::vector<size_t>> faces;
    for (std::string line; std::getline(stream, line);) {
        std::istringstream row(line);
        std::string kind; row >> kind;
        if (kind == "v") {
            double x, y, z;
            if (!(row >> x >> y >> z) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                throw std::runtime_error("Invalid vertex");
            vertices.emplace_back(x, y, z);
        } else if (kind == "f") {
            std::vector<size_t> face;
            for (std::string item; row >> item;) {
                if (item[0] == '#') break;
                item = item.substr(0, item.find('/'));
                size_t end = 0; long long index = std::stoll(item, &end);
                if (end != item.size() || !index) throw std::runtime_error("Invalid face index");
                index = index < 0 ? static_cast<long long>(vertices.size()) + index : index - 1;
                if (index < 0 || index >= static_cast<long long>(vertices.size()))
                    throw std::runtime_error("Face index out of range");
                face.push_back(static_cast<size_t>(index));
            }
            if (face.size() != 3 || face[0] == face[1] || face[1] == face[2] || face[0] == face[2])
                throw std::runtime_error("Input must contain valid triangles");
            faces.push_back(std::move(face));
        }
    }
    auto started = std::chrono::steady_clock::now();
    Mesh mesh(vertices, faces);
    mesh.setTargetTriangleCount(static_cast<size_t>(requested) * 2);
    mesh.setScaling(number("--edge-scaling", 1, 1, 4));
    mesh.setModelType(AutoRemesher::ModelType::Organic);
    mesh.setGradientAdaptivity(number("--adaptivity", 1, 0, 2));
    mesh.setAnisotropy(number("--anisotropy", 1, 0, 1));
    mesh.setSharpEdgeDegrees(number("--sharp-edge", 60, 0, 180));
    mesh.setSmoothNormalDegrees(number("--smooth-normal", 0, 0, 180));
    mesh.setProgressHandler(progress);
    if (!mesh.remesh() || mesh.remeshedVertices().empty() || mesh.remeshedQuads().empty())
        throw std::runtime_error("AutoRemesher did not produce a mesh");
    auto temporary = output; temporary += ".partial";
    size_t quads = 0, other = 0;
    try {
        std::ofstream result(temporary);
        if (!result) throw std::runtime_error("Cannot write output OBJ");
        result << std::setprecision(17);
        for (const auto& point : mesh.remeshedVertices()) {
            if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z()))
                throw std::runtime_error("Non-finite output vertex");
            result << "v " << point.x() << ' ' << point.y() << ' ' << point.z() << '\n';
        }
        for (const auto& face : mesh.remeshedQuads()) {
            if (face.size() < 3) throw std::runtime_error("Invalid output face");
            face.size() == 4 ? ++quads : ++other;
            result << 'f';
            for (auto index : face) {
                if (index >= mesh.remeshedVertices().size()) throw std::runtime_error("Invalid output index");
                result << ' ' << index + 1;
            }
            result << '\n';
        }
        result.flush();
        if (!result) throw std::runtime_error("Output write failed");
        result.close();
        std::filesystem::rename(temporary, output);
    } catch (...) {
        std::error_code error; std::filesystem::remove(temporary, error); throw;
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::ostringstream summary;
    summary << "AutoRemesher core console adapter\nQuads: " << quads << "\nNon-quads: " << other
            << "\nVertices: " << mesh.remeshedVertices().size() << "\nSeconds: " << seconds << '\n';
    for (const auto& line : mesh.phaseReport()) summary << line << '\n';
    std::cout << summary.str() << std::flush;
    if (options.count("--report")) {
        std::ofstream report(std::filesystem::u8path(options.at("--report")));
        report << summary.str();
        if (!report) throw std::runtime_error("Report write failed");
    }
    return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; ++i) args.push_back(converter.to_bytes(argv[i]));
#else
int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
#endif
    try { return run(args); }
    catch (const std::exception& error) { std::cerr << "ERROR: " << error.what() << std::endl; return 1; }
}
