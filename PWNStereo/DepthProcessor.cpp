// (c) Pawlin Technologies Ltd 2010
// purpose: basic depth map processing functions
// author: P.V. Skribtsov
// ALL RIGHTS RESERVED

#include "stdafx.h"
#include "DepthProcessor.h"

#include <PWNGeneral/pwnutil.h>
#include <PWMath/minmaxavg.h>
#include <PWNImage/colorsegment.h>
#include <PWNProfiler/duration.h>
#include <PWNColor/rgb.h>
#include <PWNGeometry/geometricobjects.h>
#include <PWNOpenCvExt/cvmathdrawingnew.h>

//#define USE_PCA
#define USE_CVSOLVE
#ifdef USE_PCA
#include <PWNCloudObj/cloudutils.h>
#endif

#ifdef USE_PROC
#include <PWMath2_0/IMatrixProcessor.h>
#endif

#include <vector>
using std::vector;

#ifdef USE_PCA
inline Point3Df computeNormal(const vector <Point3Df> &pts) {
    Eigen3Df eigen(pts, false);
    float d = eigen.abcd[3];
    if (d == 0) return Point3Df(0, 0, 0);
    return eigen.v3.multiplied(1.0f / d);
}
#endif

#ifdef USE_PROC
inline Point3Df computeNormal(const vector <Point3Df> &pts) {
    size_t maxcount = rect.height*rect.width;
    SimpleMatrix a_array(maxcount, 3);
    SimpleMatrix t_array(maxcount, 1);
    size_t samples = 0;

    float t = depth / a.z;
    float *aptr = a_array.getRow(samples);
    memcpy(aptr, &a, sizeof(float) * 3);
    t_array[samples][0] = -1.0f / t;
    samples++;

    if (samples < 3) return false;
    a_array.setSize(samples, 3);
    t_array.setSize(samples, 1);

    SimpleMatrix solution(3, 1, 0.0f);
    matprc.solve(a_array, t_array, solution, SVD);
    Point3Df resn(solution.getRow(0)[0],
        solution.getRow(0)[1], solution.getRow(0)[2]);

}
#endif

#ifdef USE_CVSOLVE
inline Point3Df computeNormal(const vector <Point3Df> &pts) {
    size_t samples = pts.size();
    cv::Mat norms((int)samples, 3, CV_32F);
    cv::Mat dinv((int)samples, 1, CV_32F);
    cv::Mat solution(3, 1, CV_32F);
    FOR_ALL(pts, i) {
        memcpy(&norms.row((int)i).at<float>(0), &pts[i], sizeof(float) * 3);
        dinv.at<float>((int)i, 0) = -1.0f;
    }
    //printf("Selection raw depth stat:\n"); depthstat.print();
    //FileSaver<Point3Df>::save("../select.3dp", cloud);
    solution = 0;
    //solution.at<float>(2) = -1.0f;
    //float v0 = (float)cv::norm(norms*solution, dinv);
    if (!cv::solve(norms, dinv, solution, cv::DECOMP_SVD)) return Point3Df(0, 0, 0);
    //float v1 = (float)cv::norm(norms*solution, dinv);
    //printf("err before opt %f, after %f\n", v0, v1);

    Point3Df resn(solution.at<float>(0), solution.at<float>(1), solution.at<float>(2));
    return resn;
}
#endif

DepthProcessor::DepthProcessor(/*IMatrixProcessor &matprc*/)// : matprc(matprc)
{
    // init default intrinsic params
    cx = 320.0f;
    cy = 240.0f;
    fx = 575.814f;
    fy = 575.814f;
}

DepthProcessor::DepthProcessor(std::string intrinsics_file) {
    //float tmp;
    std::ifstream fin3(intrinsics_file);
    if (!fin3.is_open()) throw std::runtime_error("can't open " + intrinsics_file);
    std::string s;
    std::vector<std::string> tmp_s;
    getline(fin3, s);
    split(s, '\t', tmp_s);
    fx = stof(tmp_s[0]);
    cx = stof(tmp_s[2]);
    getline(fin3, s);
    tmp_s.clear();
    split(s, '\t', tmp_s);
    fy = stof(tmp_s[1]);
    cy = stof(tmp_s[2]);
    if ((fx <= 0) || (fy <= 0))
        throw std::runtime_error("focus <= 0");
    fin3.close();
}


DepthProcessor::~DepthProcessor() {
}

bool DepthProcessor::computeNormal(float scale, const cv::Mat & img,
    const cv::Rect &rect, Point3Df &normal, float &d, const cv::Size &steps) const {
    vector <Point3Df> cloud;
    for (int i = 0; i < rect.height; i += steps.height) {
        for (int j = 0; j < rect.width; j += steps.width) {
            int x = rect.x + j;
            int y = rect.y + i;
            float depth = img.at<float>(y, x); // depth assumed in meters
            if (depth == 0) continue;
            Point3Df a = pix2ray(x*scale, y*scale);
            a.normate();
            float t = depth / a.z;
            cloud.push_back(a.multiplied(t));
        }
    }
    if (cloud.size() < 3) return false;
    Point3Df resn = ::computeNormal(cloud);

    d = 1.0f;
    float n = resn.norm();
    if (n == 0) return false;
    float k = 1.0f / n;
    d *= k;
    resn.multiply(k);
    normal = resn;

    //printf("Plane parameters:\n");
    //resn.print();
    //printf("%f\n", d);
    return true;
}
int32_t unitcode(uint32_t R, uint32_t G, uint32_t B) {
    if (R == 127 && G == 127 && B == 127) return 0;
    if (R == 0 && G == 0 && B == 0) return 0;
    return 1;
}

unsigned char charcode(uint32_t R, uint32_t G, uint32_t B) {
    return (unsigned char)unitcode(R, G, B);
}

void DepthProcessor::SegmentWholeImageWithCalculationNormal(
    const cv::Mat& in, cv::Mat& res, cv::Mat& comps, cv::Mat& depth0,
    const size_t space_step, vector <OvalBase>& ovals) const {
    //cv::imshow("source", in*0.0003);
    cv::Mat img;
    cv::Laplacian(in, img, CV_32F, 5);
    cv::Mat mask;
    cv::compare(img*0.2f, cv::Scalar(0.5f), mask, cv::CMP_LT);//0.0002f
    cv::multiply(in, mask, depth0, 1.0 / 255, CV_32F);
    //cv::imshow("laplace", mask);
    //cv::imshow("source&mask", depth0*0.3f);
    cv::waitKey(0);

    Profiler prof;
    prof.startSequence();
    float k = 1.0f / space_step;
    cv::Mat depth;
    cv::resize(depth0, depth, cv::Size(), k, k, cv::INTER_AREA);
    res = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC3); // rgb
    int wsize = depth.cols / 15; if (wsize < 2) wsize = 2;
    int step = int(wsize / 2 + 0.5f); if (step == 0) step = 1;
    printf("wsize = %d, step = %d\n", wsize, step);
    int half = wsize / 2;
    for (int i = half; i < depth.rows - half; i++) {
        unsigned char * line_out = res.row(i).data + 3 * half;

        for (int j = half; j < depth.cols - half; j++) {
            cv::Rect rect;
            rect.x = j - half, rect.y = i - half, rect.width = wsize, rect.height = wsize;
            Point3Df normal;
            float d;
            bool good = computeNormal((float)space_step, depth, rect, normal, d, cv::Size(step, step));
            //normal.print();
            //d *= 2.0f;
            //d += 1.0f;
            d = 1.0f;
            unsigned char r = 0, g = 0, b = 0;
            if (good) normal2color(normal, r, g, b);

            *line_out++ = r;
            *line_out++ = g;
            *line_out++ = b;
        }
    }
    prof.markPoint("normals compute");
    cv::medianBlur(res, res, half * 2 + 1);
    prof.markPoint("median filter");

    ColorComponentFinderX::Params params;
    //ColorComponentFinder::Params params;
    params.breakIntensityDiffThreshold = -1;
    params.breakRGBDiffThreshold = 6;
    params.check_sign = false;
    params.minpixels = wsize;
    ColorComponentFinderX finder(unitcode, params);
    //ColorComponentFinder finder(charcode, res.cols, res.rows);

    comps = cv::Mat(res.rows, res.cols, CV_32S);
    finder.segmentColor(res, ovals, &comps);
    //finder.segmentColor(&((IplImage )res), ovals, params);
    //finder.render32S(comps);
    prof.markPoint("segmentation");
    prof.print();
}

void DepthProcessor::newSegmentation(const cv::Mat& in, cv::Mat& res, cv::Mat& comps, cv::Mat& depth0, vector <OvalBase>& ovals) const {
    cv::Mat img;
    //cv::Laplacian(in, img, CV_32F, 5);
    img = in.clone();
    cv::Mat mask;
    cv::compare(img*0.0002f, cv::Scalar(0.5f), mask, cv::CMP_LT);//0.0002f
    cv::multiply(img, mask, depth0, 1.0 / 255, CV_32F); //CV_32S

    int wsize = in.cols / 15; if (wsize < 2) wsize = 2;

    ColorComponentFinderX::Params params;
    params.breakIntensityDiffThreshold = 25;
    params.breakRGBDiffThreshold = -1;
    params.check_sign = false;
    params.minpixels = wsize;
    ColorComponentFinderX finder(unitcode, params);
    comps = cv::Mat(img.rows, img.cols, CV_32S);

    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, -1, img != 0);
    img.convertTo(img, CV_8UC3, 1);

    finder.segmentColor(img, ovals, &comps); //depth0

    std::vector<cv::Vec3b> colorTab(ovals.size());
    for (int i = 0; i < ovals.size(); i++)
    {
        colorTab[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }

    res = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC3);
    for (int i = 0; i < comps.cols; i++) {
        for (int j = 0; j < comps.rows; j++)
        {
            if (comps.at<int32_t>(j, i) != -1)
                res.at<cv::Vec3b>(j, i) = colorTab[comps.at<int32_t>(j, i)];
        }
    }
}

void DepthProcessor::genChunks(
    const cv::Mat & input, cv::Mat& output, size_t space_step,
    Chunks &chunks, float outscale, bool yzchange, int mode, int dx, int dy) const {
    cv::Mat in = input.clone();

    cv::Mat comps, depth0;
    vector <OvalBase> ovals;
    if (mode == 1)
        SegmentWholeImageWithCalculationNormal(in, output, comps, depth0, space_step, ovals);
    else
        newSegmentation(in, output, comps, depth0, ovals);

    std::set <int32_t> codes;
    const float minMA = 5.0f;
    FOR_ALL(ovals, i) {
        float ma = ovals[i].getMinAxis();
        if (ma*space_step > minMA) codes.insert(ovals[i].index);
    }
    //prof.markPoint("ovals manipulation");
    std::map < int32_t, Chunks::Chunk3D > out;
    //comps.convertTo(comps, CV_32S, 1);
    meshgen.generate(*this, (float)space_step, comps, depth0, codes, out, outscale, yzchange, dx, dy);
    //prof.markPoint("mesh generation");
    //prof.print();
    chunks = Chunks(out, 5);

    //#define VISUALIZE
#ifdef VISUALIZE
    for (int y = 0; y < output.rows; y++) {
        unsigned char * line_out = output.row(y).data;
        for (int x = 0; x < res.cols; x++) {
            int yS = y / (int)space_step;
            int xS = x / (int)space_step;
            int32_t label = comps.at<int32_t>(yS, xS);
            codes.insert(label);
            unsigned char r = 255;
            unsigned char g = 255;
            unsigned char b = 0;

            *line_out++ = r;
            *line_out++ = g;
            *line_out++ = b;

        }
    }
    FOR_ALL(ovals, i) {
        int32_t label = ovals[i].index;
        //printf("oval#%d size = %f\n", label, ovals[i].getArea());
        //ovals[i].scale((float)space_step);
        pawlin::drawOvalBase(res, ovals[i], CV_RGB(255, 0, 0), true, 1);
    }
    //cv::imshow("result", res);
    //cv::waitKey(0);
#endif 

}

void DepthProcessor::genPointCloud(float scale, const cv::Mat & img, const string &filename, float k, bool yzchange, int save_every_that_point) const {
    // generate point cloud
    printf("get point cloud\n");
    vector<Point3Df> cloud;
    for (int i = 0; i < img.rows; i += save_every_that_point) {
        for (int j = 0; j < img.cols; j += save_every_that_point) {
            Point3Df p = getPoint(j, i, img, scale, k);
            if (yzchange) p = pt2vis(p);
            cloud.push_back(p);
        }
    }
    FileSaver<Point3Df>::save(filename.c_str(), cloud);
    printf("end point cloud\n");
}

void DepthProcessor::genColorPointCloud(float scale, const cv::Mat & img, const std::string & filename, float k, bool yzchange, int save_every_that_point) const {
    vector<Point3DfColor> cloud;
    cv::Mat img_8U;
    cv::normalize(img, img_8U, 0, 1, cv::NORM_MINMAX);
    img_8U.convertTo(img_8U, CV_8UC1, 255);
    for (int i = 0; i < img.rows; i += save_every_that_point) {
        for (int j = 0; j < img.cols; j += save_every_that_point) {
            int cur_depth = img_8U.at<uchar>(i, j);
            Point3Df p = getPoint(j, i, img, scale, k);
            if (yzchange) p = pt2vis(p);
            cloud.push_back(Point3DfColor(p, Point3Df(1 - (float)cur_depth / 256, 0.1f, (float)cur_depth / 256)));
        }
    }
    FileSaver<Point3DfColor>::save(filename.c_str(), cloud);

}

inline void push(Dataset &dataset, const Point3Df &p, float value) {
    dataset.rows.resize(dataset.rows.size() + 1, FloatVector(4, 0));
    memcpy(&dataset.rows.back().data[0], &p, sizeof(float) * 3);
    dataset.rows.back().data.back() = value;
}

//inline float rnd() { return rand() / (float)RAND_MAX; }
inline void checkTuple(const std::tuple < int32_t, int32_t, int32_t> &t) {
    if (std::get<0>(t) == -1)
        throw "bad";
    if (std::get<1>(t) == -1)
        throw "bad";
    if (std::get<2>(t) == -1)
        throw "bad";
}
void DepthProcessor::genVolumeNNDataset(const cv::Mat & depth32F, Dataset & dataset) const {
    const int step = 10;
    FloatVector row(4, 0);
    for (int i = 0; i < depth32F.rows; i += step) {
        for (int j = 0; j < depth32F.cols; j += step) {
            float depth = depth32F.at<float>(i, j); // depth assumed in meters
            if (depth == 0) continue;
            Point3Df a = pix2ray((float)j, (float)i);
            a.normate();
            float t = depth / a.z;
            push(dataset, a.multiplied(t), 0.5);
            push(dataset, a.multiplied(t*0.9f*rnd()), 0.0f);
            push(dataset, a.multiplied(t*(1.1f + rnd())), +1.0f);
        }
    }
}

void RasterMeshGen::generate(
    const DepthProcessor & proc,
    float codes_scale,
    const cv::Mat & codes0,
    const cv::Mat & depth,
    const set <int32_t> &allowed,
    std::map<int32_t, Chunks::Chunk3D>& out,
    float outscale, bool yzchange, int dx, int dy) const {
    //cv::Mat m(10, 10, CV_32FC3);
    //cv::Vec3f elem = m.at<cv::Vec3f>(1, 1);
    cv::Mat codes = codes0.clone();
    cv::Mat indices(codes.rows, codes.cols, CV_32S);
    indices = -1;
    MinMaxAvg pstat;
    int maxx = codes.cols;
    int maxy = codes.rows;
    for (int y = 0; y < maxy; y++) {
        for (int x = 0; x < maxx; x++) {
            int32_t code = codes.at<int32_t>(y, x);
            if (allowed.count(code) == 0) continue; // oval was not generated for this code
            Point3Df p = proc.getPoint(int(x*codes_scale + dx), int(y*codes_scale + dy), depth, 1.0f, outscale, dx, dy);
            if (p.norm() == 0.0f) {
                codes.at<int32_t>(y, x) = -1;
                continue;
            }
            if (yzchange) p = Point3Df(p.x, p.z, -p.y);
            pstat.take(p.norm());
            size_t index = out[code].points.size();
            out[code].points.push_back(p);
            indices.at<int32_t>(y, x) = (int32_t)index;
        }
    }
    pstat.print(true);
    // recompute normals and project points to computed plane
    for (auto iter = out.begin(); iter != out.end(); iter++) {
        Chunks::Chunk3D &chunk = iter->second;
        chunk.d = 1.0f;
        if (chunk.points.size() >= 3) {
            chunk.normal = computeNormal(chunk.points);
            //FOR_ALL(chunk.points, i) {
            //	chunk.points[i] = project(chunk.normal, chunk.d, chunk.points[i]);
            //}
        } else chunk.normal = Point3Df(1.0f, 0, 0);
    }
    // generate mesh triangles for each code
    int32_t abcd[4];
    for (int y = 0; y < codes.rows - 1; y++) {
        for (int x = 0; x < codes.cols - 1; x++) {
            abcd[0] = codes.at<int32_t>(y, x);
            abcd[1] = codes.at<int32_t>(y, x + 1);
            abcd[2] = codes.at<int32_t>(y + 1, x);
            abcd[3] = codes.at<int32_t>(y + 1, x + 1);
            for (int n = 0; n < 4; n++) {
                int32_t current_code = abcd[n];
                bool found = false;
                for (int k = 0; k < n; k++) if (current_code == abcd[k]) { found = true; break; }
                if (found || allowed.count(current_code) == 0) continue;
                char hash = 0;
                char bit = 8;
                for (int n = 0; n < 4; n++, bit /= 2) if (abcd[n] == current_code) hash |= bit;
                const string &str = v.find(hash)->second;
                Variant var(str);
                auto &mesh = out[current_code].mesh;
                try {
                    if (var.abc) {
                        mesh.push_back(
                            std::make_tuple(
                                indices.at<int32_t>(y + 0, x + 0),
                                indices.at<int32_t>(y + 0, x + 1),
                                indices.at<int32_t>(y + 1, x + 0)
                            )
                        );
                        checkTuple(mesh.back());
                    }
                    if (var.acd) {
                        mesh.push_back(
                            std::make_tuple(
                                indices.at<int32_t>(y + 0, x + 0),
                                indices.at<int32_t>(y + 1, x + 1),//
                                indices.at<int32_t>(y + 1, x + 0)
                            )
                        );
                        checkTuple(mesh.back());
                    }
                    if (var.bad) {
                        mesh.push_back(
                            std::make_tuple(
                                indices.at<int32_t>(y + 0, x + 0),
                                indices.at<int32_t>(y + 1, x + 1),
                                indices.at<int32_t>(y + 0, x + 1) //
                            )
                        );
                        checkTuple(mesh.back());
                    }
                    if (var.bcd) {
                        mesh.push_back(
                            std::make_tuple(
                                indices.at<int32_t>(y + 0, x + 1),//b
                                indices.at<int32_t>(y + 1, x + 1),//d
                                indices.at<int32_t>(y + 1, x + 0) //c
                            )
                        );
                        checkTuple(mesh.back());
                    }
                }
                catch (...) {
                    for (int n = 0; n < 4; n++) printf("%d ", abcd[n]);
                    printf(" cur_code %d\n", current_code);
                }
            }
        }
    }
}

void NeuralRayTracer::genNNSurfaceCloud(float scale, const cv::Size & sz, const DepthProcessor & dproc, float k) const {
    const int step = 2;
    const float maxd = 5.0f;
    const float dstep = 0.01f;
    const float thresh = 0.5f;
    // generate point cloud
    vector<Point3DfTemp> cloud;
    vector<float> storage(net.getStorageSize());
    for (int i = 0; i < sz.height; i += step) {
        for (int j = 0; j < sz.width; j += step) {
            bool found = false;
            Point3Df s;
            for (float d = 0; d < maxd; d += dstep) {
                s = dproc.getPoint(j, i, d, scale, 1.0f);
                float v = f(s, storage);
                if (v > thresh) {
                    cloud.push_back(Point3DfTemp(dproc.pt2vis(s)*k, 1.0f - d / maxd));
                    break;
                }
            }
        }
    }
    const string filename = "nnout.3dip";
    FileSaver<Point3DfTemp>::save(filename.c_str(), cloud);

}
