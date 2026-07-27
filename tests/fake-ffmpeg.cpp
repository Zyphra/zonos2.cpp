#include <fcntl.h>
#include <io.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::vector<std::string> expected = {
        "-nostdin", "-v", "error", "-i", "", "-ac", "1",
        "-ar", "24000", "-f", "f32le", "-",
    };
    if (argc != (int) expected.size() + 1) {
        fprintf(stderr, "fake-ffmpeg: expected %zu arguments, got %d\n", expected.size(), argc - 1);
        return 2;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (i == 4) continue;
        if (argv[i + 1] != expected[i]) {
            fprintf(stderr, "fake-ffmpeg: unexpected argument %zu: %s\n", i + 1, argv[i + 1]);
            return 2;
        }
    }
    if (!std::filesystem::exists(std::filesystem::u8path(argv[5]))) {
        fprintf(stderr, "fake-ffmpeg: input path was not preserved: %s\n", argv[5]);
        return 2;
    }
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
        fprintf(stderr, "fake-ffmpeg: cannot set binary stdout\n");
        return 2;
    }

    const std::vector<float> pcm(12000, 0.25f);
    if (fwrite(pcm.data(), sizeof(float), pcm.size(), stdout) != pcm.size()) {
        fprintf(stderr, "fake-ffmpeg: cannot write output\n");
        return 2;
    }
    return 0;
}
