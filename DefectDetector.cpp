#include "DefectDetector.h"
static int obj_color[36][3]
{
    {64, 0, 128}, {14, 201, 255}, {0, 191, 72},
    {128, 128, 255}, {0, 128, 0}, {164, 73, 163},
    {64, 0, 64}, {100, 180, 180}, {163, 73, 164},
    {255, 200, 15}, {128, 0, 255}, {90, 109, 165},
    {0, 162, 232}, {165, 162, 27}, {200, 145, 255},
    {20, 30, 130}, {255, 128, 192}, {0, 128, 128},
    {128, 0, 64}, {128, 0, 0}, {150, 0, 150},
    {0, 0, 128}, {200,200,0}, {90,113,15},
    {6,121,101}, {75,4,123}, {128, 0, 128},
    {18,62,109}, {0, 64, 64}, {97,20,107},
    {69,44,84}, {128, 128, 192}, {128,128,255},
    {128,64,128}, {50, 60, 220}, {0, 64, 128}
};



void DefectDetector::filterContours(std::vector<std::vector<cv::Point>> *contours) {
    while (contours->size() > 5) {
        double min_area = DBL_MAX;
        int min_it = 0;
        for (size_t i = 0; i < contours->size(); i++)
        {
            if (contourArea((*contours)[i]) < min_area) {
                min_area = contourArea((*contours)[i]);
                min_it = i;
            }
        }
        contours->erase(contours->begin() + min_it);
    }
}


void DefectDetector::showDistance(int action, int x, int y, int flags, void *userdata) {
    DataForCalculations *data = (DataForCalculations *)userdata;
    if (action == cv::EVENT_LBUTTONDOWN)
    {
        if (data->points.empty()) {
            data->points.push_back(data->proc.getPoint(x, y, data->image.at<float>(y, x)));
        } else {
            Point3Df other_point = data->proc.getPoint(x, y, data->image.at<float>(y, x));
            float dist = other_point.distance(data->points[0]);
            data->points.clear();
            putText(data->image, std::to_string(dist), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
        }
    } else if (action == cv::EVENT_RBUTTONDOWN) {
        std::cout << "ooooo\n";
        float z_dist = data->image.at<float>(y, x);
        putText(data->image, std::to_string(z_dist), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    }
}

std::string DefectDetector::numberToText(float number) {
    std::string str_number = std::to_string(number);
    auto iter = str_number.find_last_of(".");
    if (iter != std::string::npos) {
        str_number = str_number.substr(0, iter + 2);
    }
    return str_number;
}

bool DefectDetector::printDistance(int &it, cv::Rect bounding_rect, DepthProcessor &proc, float plane_min_point,/* float med_dist,*/ float contour_area, cv::Mat &contour_mask, cv::Mat &image_for_dist, cv::Mat &to_show) {

    rectangle(to_show, bounding_rect, cv::Scalar::all(70), 2);
    int x = bounding_rect.x;
    int y = bounding_rect.y;
    int h = bounding_rect.height;
    int w = bounding_rect.width;
    int pixel_square = h * w;

    Point3Df first_ver_point = proc.getPoint(x, y, image_for_dist.at<float>(y, x) + plane_min_point);
    Point3Df second_ver_point = proc.getPoint(x, y + h, image_for_dist.at<float>(y + h, x) + plane_min_point);
    Point3Df second_hor_point = proc.getPoint(x + w, y, image_for_dist.at<float>(y, x + w) + plane_min_point);

    float hor_dist = first_ver_point.distancexy(second_hor_point);
    float ver_dist = first_ver_point.distancexy(second_ver_point);
    float coef = (hor_dist*ver_dist) / (float)pixel_square;
    float real_square = contour_area * coef;
    double min_dep, max_dep;
    cv::Point min_point, max_point;
    minMaxLoc(image_for_dist, &min_dep, &max_dep, &min_point, &max_point, contour_mask);
    //if (min_dep > -0.03f) return false;
    const int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_coeff = 1.8;
    cv::Scalar cur_color = cv::Scalar(obj_color[it][0], obj_color[it][1], obj_color[it][2]);
    putText(to_show, "length=" + numberToText(ver_dist * 100) + "cm", cv::Point(642, (it * 55) + 17), font_face, 0.3 *font_coeff, cv::Scalar::all(0), 1, cv::LINE_AA);
    putText(to_show, "width=" + numberToText(hor_dist * 100) + "cm", cv::Point(642, (it * 55) + 34), font_face, 0.3 *font_coeff, cv::Scalar::all(0), 1, cv::LINE_AA);
    //putText(to_show, "square=" + numberToText(real_square * 10000) + "cm^2", cv::Point(642, (it * 55) + 34),font_face, 0.3 *font_coeff, cv::Scalar::all(0), font_thickness, cv::LINE_AA);
    putText(to_show, "depth=" + numberToText((min_dep) * -100) + "cm", cv::Point(642, (it * 55) + 51), font_face, 0.3 *font_coeff, cv::Scalar::all(0), 1, cv::LINE_AA);
    putText(to_show, "length=" + numberToText(ver_dist * 100) + "cm", cv::Point(641, (it * 55) + 17), font_face, 0.3 *font_coeff, cur_color, 1, cv::LINE_AA);
    putText(to_show, "width=" + numberToText(hor_dist * 100) + "cm", cv::Point(641, (it * 55) + 34), font_face, 0.3 *font_coeff, cur_color, 1, cv::LINE_AA);
    //putText(to_show, "square=" + numberToText(real_square * 10000) + "cm^2", cv::Point(641, (it * 55) + 34), font_face, 0.3 *font_coeff, cur_color, font_thickness, cv::LINE_AA);
    putText(to_show, "depth=" + numberToText((min_dep) * -100) + "cm", cv::Point(641, (it * 55) + 51), font_face, 0.3 *font_coeff, cur_color, 1, cv::LINE_AA);
    ++it;
    return true;
}

void DefectDetector::getImageWithMeasures(std::vector<std::vector<cv::Point>> &contours, cv::Mat &gray_copy, float min_point, bool &hole_exist, cv::Mat &gray_image_for_showing) {
    int area = 500;
    double con_area;
    DepthProcessor proc;
    proc.setIntrinsic(420.f, 420.f, 320.f, 240.f);
    gray_image_for_showing.setTo(cv::Scalar(128, 128, 128));
    cv::Mat cols = cv::Mat::zeros(cv::Size(200, 480), CV_8UC3);
    cols.setTo(cv::Scalar(200, 200, 200));
    cv::hconcat(gray_image_for_showing, cols, gray_image_for_showing);
    int actual_counter = 0;
    hole_exist = false;
    for (int i = 0; i < contours.size(); i++) {
        con_area = contourArea(contours[i]);
        if (con_area < area) continue;
        cv::Mat contour_mask = cv::Mat::zeros(gray_copy.size(), CV_8UC1);
        drawContours(contour_mask, contours, i, (255), cv::FILLED);
        cv::Scalar cur_color = cv::Scalar(obj_color[actual_counter][0], obj_color[actual_counter][1], obj_color[actual_counter][2]);
        bool found;
        found = printDistance(actual_counter, boundingRect(contours[i]), proc, min_point, con_area, contour_mask, gray_copy, gray_image_for_showing);
        if (found) {
            drawContours(gray_image_for_showing, contours, i, cur_color, cv::FILLED);
            hole_exist = true;
        }
    }
    cv::line(gray_image_for_showing, cv::Point(640 - 1, 0), cv::Point(640 - 1, 480), cv::Scalar::all(255), 2, cv::LINE_AA);
}

DefectDetector::DefectDetector(ArgParser & parser) : parser(parser), app_params_(parser.getString("app_params", "")) {
}

void DefectDetector::run() {
    switch (app_params_.mode) {
    case 0: //process files and get reports
        loadAndProcessAllFiles();
        break;
    default:
        printf("Unexpected mode, check the configs.\n");
    }
}

void DefectDetector::loadAndProcessAllFiles() {
    std::vector<std::vector<Point3Df>> pts_story;
    FileList file_proc2("/home/", "depth*.png");
    std::vector<std::string> file_names = file_proc2.getFiles();
    for (auto file_name : file_names) {
        proccesFile(file_name);
    }
}

void DefectDetector::proccesFile(std::string filename) {
    ImagePreparator IP(filename, app_params_);
    cv::Mat depth, dif_depth;
    Plane3Df plane;
    IP.getPostprocessedDepth(depth);
    IP.getPlane(plane);
    //float cloud_min = 100, cloud_max = -100;
    float min_point_in_plane = IP.getDifferenceDepth(dif_depth);
    cv::Mat new_mask = dif_depth < -0.01f;
    new_mask.convertTo(new_mask, CV_8UC1);
    std::vector<std::vector<cv::Point>> new_contours;
    std::vector<cv::Vec4i> new_hierarchy;
    findContours(new_mask, new_contours, new_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    filterContours(&new_contours);
    bool found_hole = true;
    cv::Mat gray_image_for_showing = cv::Mat::zeros(dif_depth.size(), CV_8UC3);
    getImageWithMeasures(new_contours, dif_depth, min_point_in_plane, found_hole, gray_image_for_showing);
    cv::Mat original_distances = dif_depth.clone();
    cv::normalize(dif_depth, dif_depth, 1, 0, cv::NORM_MINMAX);
    dif_depth.convertTo(dif_depth, CV_8UC1, 255);

    std::size_t start_pos = filename.find('_');
    std::size_t end_pos = filename.find('.');
    std::string pic_number = filename.substr(start_pos, end_pos - start_pos);
    cv::imwrite("measures15" + pic_number + ".jpg", gray_image_for_showing);
    cv::imwrite("gray_image" + pic_number + ".jpg", depth);
}
