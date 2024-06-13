#pragma once
#include <PWNGeneral/PWNCvSettings_Linux.h>

struct AppParams {
    DECLARE_PARAM(int, verboseLevel, AppParams);
    DECLARE_PARAM(int, mode, AppParams);
    DECLARE_PARAM(float, depth_units, AppParams);
    DECLARE_PARAM(int, width, AppParams);
    DECLARE_PARAM(int, height, AppParams);
    DECLARE_PARAM(int, pts_per_frame, AppParams);

    PARAM_SAVELOAD(AppParams);

    AppParams(const std::string &params_file = "./app_configs.yml") {
        mode = 0;
        depth_units = 0.001;
        width = 640;
        height = 480;
        pts_per_frame = 5;
        if (params_file != "") {
            load(params_file);
        }
    }
};
