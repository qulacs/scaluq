#include <optional>
#include <ranges>
#include <sstream>
#include <variant>
#include <vector>

class DocString {
public:
    using Code = std::vector<std::string_view>;
    using Block = std::variant<std::string_view, Code>;
    using Blocks = std::vector<Block>;
    struct Arg {
        std::string_view name;
        std::string_view type;
        bool is_optional;
        Blocks description;
    };
    struct Ret {
        std::string_view type;
        Blocks description;
    };

    explicit DocString() {}
    DocString& desc(const Block& desc) {
        description.push_back(desc);
        return *this;
    }
    DocString& arg(const Arg& arg) {
        args.push_back(arg);
        return *this;
    }
    DocString& ret(const Ret& ret) {
        returns = ret;
        return *this;
    }
    DocString& ex(const Block& ex) {
        examples.push_back(ex);
        return *this;
    }
    DocString& note(const Block& note) {
        notes.push_back(note);
        return *this;
    }

    std::string build_as_google_style() {
        std::ostringstream out;
        auto output = [&](const Block& b, std::string_view indent) {
            if (b.index() == 0) {
                out << indent << std::get<0>(b) << "\n\n";
            } else {
                const auto& code = std::get<1>(b);
                for (const auto& line : code) {
                    out << indent << line << "\n";
                }
                out << "\n";
            }
        };
        for (const auto& b : description) {
            output(b, "");
        }
        if (!args.empty()) {
            out << "Args:\n";
            for (const auto& arg : args) {
                out << "    " << arg.name << " (" << arg.type
                    << (arg.is_optional ? ", optional):\n" : "):\n");
                for (const auto& b : arg.description) {
                    output(b, "        ");
                }
            }
        }
        if (returns.has_value()) {
            out << "Returns:\n";
            const auto& ret = returns.value();
            out << "    " << ret.type << ":\n";
            for (const auto& b : ret.description) {
                output(b, "        ");
            }
        }
        if (!examples.empty()) {
            out << "Examples:\n";
            for (const auto& b : examples) {
                output(b, "    ");
            }
        }
        if (!notes.empty()) {
            out << "Notes:\n";
            for (const auto& b : notes) {
                output(b, "    ");
            }
        }
        return out.str();
    }

private:
    Blocks description;
    std::vector<Arg> args;
    std::optional<Ret> returns;
    Blocks examples;
    Blocks notes;
};
