#include "DefectDetector.h"

int main(int argc, char *argv[]) {
    ArgParser parser(argc, argv);
    DefectDetector app(parser);
    app.run();
    return 0;
}