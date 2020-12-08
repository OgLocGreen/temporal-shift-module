#include <iostream>
#include <string>
#include <tuple>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <queue>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <dnndk/n2cube.h>

#include "helper.h"

std::string KINETICS_PATH = "../kinetics_test/air_drumming/--nQbRBEz2s_104/";
std::string arg_path = "";

const bool HEADLESS = false;
std::string WINDOW_NAME = "TSM on FPGA";

const bool USE_SOFTMAX = false;

// mode for dpuCreateTask (T_MODE_NORMAL, T_MODE_PROF, T_MODE_DEBUG
#define TASK_MODE T_MODE_NORMAL
#define NUM_KERNELS 17
#define BATCH_SIZE 8
bool DEVICE_OPEN = false;

using std::chrono::high_resolution_clock;
template <typename T>
auto to_ms(T const& duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void fail(std::string err) {
    std::cerr << err << std::endl;

    if (DEVICE_OPEN)
        dpuClose();

    exit(1);
}

struct Node {
    std::string name; // "" marks invalid node
    int shape[3]; // HWC

    size_t size() const { return BATCH_SIZE*shape[0]*shape[1]*shape[2]; }
    size_t batchSize() const { return shape[0]*shape[1]*shape[2]; }
};

struct TSMSplit {
    std::string kernelName;
    Node shifted_input_node;
    Node shortcut_input_node;
    Node output_node;

    int8_t* virt_in = nullptr;
    int8_t* phys_in = nullptr;
    int8_t* virt_out = nullptr;
    int8_t* phys_out = nullptr;

    void* inHandle = nullptr;
    void* outHandle = nullptr;

    int8_t* shiftedAddr() { return virt_in; }
    int8_t* shortcutAddr() { return virt_in + shifted_input_node.size(); }
    int8_t* outAddr()   { return virt_out; }

    size_t inSize() { return shifted_input_node.size() + shortcut_input_node.size(); }
    size_t outSize() { return output_node.size(); }
};

// Buffer for non-split-IO first stage
int8_t* inSplitBuffer;

// list of kernel information:
// {Kernel Name,
//      residual tensor {name, shape}
//      shifted tensor {name, shape}
//      output tensor {name, shape}
TSMSplit splitInfo[NUM_KERNELS] = {
    {"tsm_resnet50_8f_0_1",
        {"in_imgs:0", {224,224,3}},
        {"", {0,0,0}},
        {"resnet_v1_50_pool1_MaxPool:0", {56,56,64}}},
    {"tsm_resnet50_8f_1_1",
        // Shifted input
        {"resnet_v1_50_block1_unit_1_bottleneck_v1_conv_input:0", {56,56,64}},
        // Shortcut Input
        {"resnet_v1_50_block1_unit_1_bottleneck_v1_shortcut_input:0", {56,56,64}},
        // Output
        {"resnet_v1_50_block1_unit_1_bottleneck_v1_Relu:0", {56,56,256}}},
    {"tsm_resnet50_8f_1_2",
        {"resnet_v1_50_block1_unit_2_bottleneck_v1_conv_input:0",      {56,56,256}},
        {"resnet_v1_50_block1_unit_2_bottleneck_v1_shortcut_input:0",  {56,56,256}},
        {"resnet_v1_50_block1_unit_2_bottleneck_v1_Relu:0",            {56,56,256}}},
    {"tsm_resnet50_8f_1_3",
        {"resnet_v1_50_block1_unit_3_bottleneck_v1_conv_input:0",     {56,56,256}},
        {"resnet_v1_50_block1_unit_3_bottleneck_v1_shortcut_input:0", {56,56,256}},
        {"resnet_v1_50_block1_unit_3_bottleneck_v1_Relu:0",           {28,28,256}}},
    {"tsm_resnet50_8f_2_1",
        {"resnet_v1_50_block2_unit_1_bottleneck_v1_conv_input:0",     {28,28,256}},
        {"resnet_v1_50_block2_unit_1_bottleneck_v1_shortcut_input:0", {28,28,256}},
        {"resnet_v1_50_block2_unit_1_bottleneck_v1_Relu:0",           {28,28,512}}},
    {"tsm_resnet50_8f_2_2",
        {"resnet_v1_50_block2_unit_2_bottleneck_v1_conv_input:0",     {28,28,512}},
        {"resnet_v1_50_block2_unit_2_bottleneck_v1_shortcut_input:0", {28,28,512}},
        {"resnet_v1_50_block2_unit_2_bottleneck_v1_Relu:0",           {28,28,512}}},
    {"tsm_resnet50_8f_2_3",
        {"resnet_v1_50_block2_unit_3_bottleneck_v1_conv_input:0",     {28,28,512}},
        {"resnet_v1_50_block2_unit_3_bottleneck_v1_shortcut_input:0", {28,28,512}},
        {"resnet_v1_50_block2_unit_3_bottleneck_v1_Relu:0",           {28,28,512}}},
    {"tsm_resnet50_8f_2_4",
        {"resnet_v1_50_block2_unit_4_bottleneck_v1_conv_input:0",     {28,28,512}},
        {"resnet_v1_50_block2_unit_4_bottleneck_v1_shortcut_input:0", {28,28,512}},
        {"resnet_v1_50_block2_unit_4_bottleneck_v1_Relu:0",           {14,14,512}}},
    {"tsm_resnet50_8f_3_1",
        {"resnet_v1_50_block3_unit_1_bottleneck_v1_conv_input:0",     {14,14,512}},
        {"resnet_v1_50_block3_unit_1_bottleneck_v1_shortcut_input:0", {14,14,512}},
        {"resnet_v1_50_block3_unit_1_bottleneck_v1_Relu:0",           {14,14,1024}}},
    {"tsm_resnet50_8f_3_2",
        {"resnet_v1_50_block3_unit_2_bottleneck_v1_conv_input:0",     {14,14,1024}},
        {"resnet_v1_50_block3_unit_2_bottleneck_v1_shortcut_input:0", {14,14,1024}},
        {"resnet_v1_50_block3_unit_2_bottleneck_v1_Relu:0",           {14,14,1024}}},
    {"tsm_resnet50_8f_3_3",
        {"resnet_v1_50_block3_unit_3_bottleneck_v1_conv_input:0",     {14,14,1024}},
        {"resnet_v1_50_block3_unit_3_bottleneck_v1_shortcut_input:0", {14,14,1024}},
        {"resnet_v1_50_block3_unit_3_bottleneck_v1_Relu:0",           {14,14,1024}}},
    {"tsm_resnet50_8f_3_4",
        {"resnet_v1_50_block3_unit_4_bottleneck_v1_conv_input:0",     {14,14,1024}},
        {"resnet_v1_50_block3_unit_4_bottleneck_v1_shortcut_input:0", {14,14,1024}},
        {"resnet_v1_50_block3_unit_4_bottleneck_v1_Relu:0",           {14,14,1024}}},
    {"tsm_resnet50_8f_3_5",
        {"resnet_v1_50_block3_unit_5_bottleneck_v1_conv_input:0",     {14,14,1024}},
        {"resnet_v1_50_block3_unit_5_bottleneck_v1_shortcut_input:0", {14,14,1024}},
        {"resnet_v1_50_block3_unit_5_bottleneck_v1_Relu:0",           {14,14,1024}}},
    {"tsm_resnet50_8f_3_6",
        {"resnet_v1_50_block3_unit_6_bottleneck_v1_conv_input:0",     {14,14,1024}},
        {"resnet_v1_50_block3_unit_6_bottleneck_v1_shortcut_input:0", {14,14,1024}},
        {"resnet_v1_50_block3_unit_6_bottleneck_v1_Relu:0",           {7,7,1024}}},
    {"tsm_resnet50_8f_4_1",
        {"resnet_v1_50_block4_unit_1_bottleneck_v1_conv_input:0",     {7,7,1024}},
        {"resnet_v1_50_block4_unit_1_bottleneck_v1_shortcut_input:0", {7,7,1024}},
        {"resnet_v1_50_block4_unit_1_bottleneck_v1_Relu:0",           {7,7,2048}}},
    {"tsm_resnet50_8f_4_2",
        {"resnet_v1_50_block4_unit_2_bottleneck_v1_conv_input:0",     {7,7,2048}},
        {"resnet_v1_50_block4_unit_2_bottleneck_v1_shortcut_input:0", {7,7,2048}},
        {"resnet_v1_50_block4_unit_2_bottleneck_v1_Relu:0",           {7,7,2048}}},
    {"tsm_resnet50_8f_4_3",
        {"resnet_v1_50_block4_unit_3_bottleneck_v1_conv_input:0",     {7,7,2048}},
        {"resnet_v1_50_block4_unit_3_bottleneck_v1_shortcut_input:0", {7,7,2048}},
        {"Linear_MatMul:0",                                           {1,1,400}}}
};


void memcpy_assign(void* dst, void* src, size_t sz) {
    char* cdst = (char*)dst;
    char* csrc = (char*)src;
    for (size_t i = 0; i < sz; i++) {
        cdst[i] = csrc[i];
    }
}

void memset_assign(void* dst, int val, size_t sz) {
    char* cdst = (char*)dst;
    for (size_t i = 0; i < sz; i++) {
        cdst[i] = val;
    }
}

void allocBuffers(DPUTask* tasks[NUM_KERNELS]) {
    inSplitBuffer = (int8_t*)malloc(splitInfo[0].output_node.size());
    if (!inSplitBuffer)
        fail("Failed to allocated buffer");

    for (int i = 1; i < NUM_KERNELS; i++) {
        // TODO: Implement custom alloc
        // Allocate input buffers {residual, shifted}
        splitInfo[i].inHandle = dpuAllocMem(splitInfo[i].inSize(),
                (int8_t*&)splitInfo[i].virt_in, (int8_t*&)splitInfo[i].phys_in);
        splitInfo[i].outHandle = dpuAllocMem(splitInfo[i].outSize(),
                (int8_t*&)splitInfo[i].virt_out, (int8_t*&)splitInfo[i].phys_out);

        if(!splitInfo[i].inHandle)
            fail("Failed to allocated split input buffer");
        if(!splitInfo[i].outHandle)
            fail("Failed to allocated split output buffer");

        dpuBindInputTensorBaseAddress(tasks[i],
                (int8_t*&)splitInfo[i].virt_in, (int8_t*&)splitInfo[i].phys_in);
        dpuBindOutputTensorBaseAddress(tasks[i],
                (int8_t*&)splitInfo[i].virt_out, (int8_t*&)splitInfo[i].phys_out);

        //printf("IN: %p, %p\n", splitInfo[i].virt_in, splitInfo[i].phys_in);
        //printf("OUT: %p, %p\n", splitInfo[i].virt_out, splitInfo[i].phys_out);
    }
}

void freeBuffers() {
    free(inSplitBuffer);

    for (int i = 1; i < NUM_KERNELS; i++) {
        dpuFreeMem(splitInfo[i].inHandle);
        dpuFreeMem(splitInfo[i].outHandle);
    }
}

int8_t* input_cmp[NUM_KERNELS] = {0};

void rescale_input(int split_num, float scale, DPUTask* tasks[NUM_KERNELS]) {
    const Node& input_node = splitInfo[split_num - 1].output_node;
    int8_t* input_data = splitInfo[split_num - 1].outAddr();
    if (split_num - 1 == 0)
        input_data = inSplitBuffer;
        //input_data = dpuGetTensorAddress(dpuGetBoundaryIOTensor(tasks[0], splitInfo[0].output_node.name.c_str()));

    for (int s = 0; s < BATCH_SIZE; s++) {
        for (int i = 0; i < input_node.shape[0]; i++) {
            for (int j = 0; j < input_node.shape[1]; j++) {
                for (int k = 0; k < input_node.shape[2]; k++) {
                    input_data[s*input_node.batchSize()
                               + i*input_node.shape[1]*input_node.shape[2]
                               + j*input_node.shape[2] + k] *= scale;
                }
            }
        }
    }
}


void doShift(int split_num, DPUTask* tasks[NUM_KERNELS]) {
    assert(split_num > 0);

    const Node& input_node = splitInfo[split_num - 1].output_node;
    int8_t* input_data = splitInfo[split_num - 1].outAddr();
    if (split_num - 1 == 0)
        input_data = inSplitBuffer;

    const Node& output_node = splitInfo[split_num].shifted_input_node;
    int8_t* output_data = splitInfo[split_num].shiftedAddr();

    int c = input_node.shape[2] / 8;

    for (int segment = 0; segment < BATCH_SIZE; segment++) {
        for (int h = 0; h < input_node.shape[0]; h++) {
            for (int w = 0; w < input_node.shape[1]; w++) {
                int8_t* in = input_data + segment*input_node.batchSize()
                                        + h*input_node.shape[1]*input_node.shape[2]
                                        + w*input_node.shape[2];

                // Current segment for unshifted values
                int8_t* out_cur = output_data + segment*input_node.batchSize()
                                              + h*input_node.shape[1]*input_node.shape[2]
                                              + w*input_node.shape[2];


                // Shift to previous segment if not first segment
                int8_t* out_prev = nullptr;
                if (segment > 0) {
                    out_prev = output_data + (segment-1)*input_node.batchSize()
                                               + h*input_node.shape[1]*input_node.shape[2]
                                               + w*input_node.shape[2];
                }

                // Shift to next segment if not last segment
                int8_t* out_next = nullptr;
                if (segment < BATCH_SIZE-1) {
                    out_next = output_data + (segment+1)*input_node.batchSize()
                                               + h*input_node.shape[1]*input_node.shape[2]
                                               + w*input_node.shape[2];
                }


                // shift to previous
                if (out_prev) {
                    memcpy_assign(out_prev, in, c*sizeof(int8_t));
                }

                // shift to next
                if (out_next) {
                    memcpy_assign(out_next + c, in + c, c*sizeof(int8_t));
                }

                // copy remaining channels to same segment
                memcpy_assign(out_cur + 2*c, in + 2*c, (input_node.shape[2] - 2*c)*sizeof(int8_t));

                // pad zero on first/last segment
                if (segment == 0) {
                    memset_assign(out_cur + c, 0, c*sizeof(int8_t));
                } else if (segment == BATCH_SIZE - 1) {
                    memset_assign(out_cur, 0, c*sizeof(int8_t));
                }
            }
        }
    }

    // TODO: Merge residual and input split memory regions
    memcpy_assign(splitInfo[split_num].shortcutAddr(), input_data, input_node.size());
}

void runTSMSerial(DPUTask* tasks[NUM_KERNELS]) {
    float prev_out_scale;
    for (int i = 0; i < NUM_KERNELS; i++) {

        //// SCALING
        float out_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[i], splitInfo[i].output_node.name.c_str()));
        float shift_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[i], splitInfo[i].shifted_input_node.name.c_str()));

        float shortcut_scale = 0;

        if (i > 0) {
            shortcut_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[i], splitInfo[i].shortcut_input_node.name.c_str()));

            // In all scales a costant scale is applied across all inputs
            assert(shift_scale == shortcut_scale);

            // Rescale input if previous output scaling doesn't cancel current input scaling
            // i.e. prev_out_scale * in_scale != 1
            float rescale = prev_out_scale * shift_scale;
            if (rescale != 1) {
                //printf("RESCALE %d (%f) %f, %f\n", i, rescale, prev_out_scale, shift_scale);
                rescale_input(i, rescale, tasks);
            }


            //// SHIFTING
            auto t1 = high_resolution_clock::now();
            doShift(i, tasks);
            auto t2 = high_resolution_clock::now();
            //printf("shift %d = %d\n", i, (t2-t1).count());
        }


        //// TASK RUNNING
        if (i > 0)
            dpuSyncMemToDev(splitInfo[i].inHandle, 0, splitInfo[i].inSize());
        printf("Running task [%d]\n", i);
        dpuRunTask(tasks[i]);

        if (i > 0)
            dpuSyncDevToMem(splitInfo[i].outHandle, 0, splitInfo[i].outSize());

        if (i == 0)
            memcpy(inSplitBuffer, dpuGetTensorAddress(dpuGetBoundaryIOTensor(tasks[0], splitInfo[0].output_node.name.c_str())), splitInfo[0].output_node.size());
            //dpuGetOutputTensorInHWCInt8(tasks[0], splitInfo[0].output_node.name.c_str(), inSplitBuffer, splitInfo[0].output_node.size());
        //printf("[%d] out (%p): %d, %f\n", i, splitInfo[i].outAddr(), *splitInfo[i].outAddr(), (*splitInfo[i].outAddr())*out_scale);

        prev_out_scale = out_scale;
        //printf("[%d] shift_scale: %f, resid_scale: %f, outscale: %f\n", i, shift_scale, resid_scale, out_scale);
    }
}

// TODO: Fill in labels
std::string categories[400];

int processOutput(int raw_gesture, float features[400]) {
    const int HISTORY_FEAT_LEN = 12;
    const int HISTORY_LEN = 20;

    static int i_logit = 0;
    static std::vector<std::array<float, 27>> history_feat(HISTORY_FEAT_LEN, std::array<float, 27>{});
    static std::deque<int> history = {2, 2};

    // Copy round of features into running feature history array
    std::copy_n(features, 27, history_feat[i_logit].data());
    i_logit++;
    if (++i_logit >= 12)
        i_logit = 0;

    // Get current gesture across length of feature history
    std::array<float, 27> sums{};
    for (int i = 0; i < HISTORY_FEAT_LEN; i++) {
        for (int j = 0; j < 27; j++) {
            sums[j] += history_feat[i][j];
        }
    }

    int gesture = std::distance(sums.begin(), std::max_element(sums.begin(), sums.end()));

	if (gesture == 0)
		gesture = 2;

    // Apply history smoothing
    if (gesture != *history.rbegin() && *history.rbegin() != *(history.rbegin() + 1))
        gesture = *history.rbegin();

    history.push_back(gesture);
    if (history.size() > HISTORY_LEN)
        history.pop_front();

    return *history.rbegin();
}

int run(cv::VideoCapture& cap) {
    DPUKernel* kernels[NUM_KERNELS];
    DPUTask*   tasks[NUM_KERNELS];

    if (dpuOpen() != 0)
        fail("Error opening DPU device");

    DEVICE_OPEN = true;

    for (int i = 0; i < NUM_KERNELS; i++) {
        kernels[i] = dpuLoadKernel(splitInfo[i].kernelName.c_str());
        tasks[i] = dpuCreateTask(kernels[i], TASK_MODE);
    }

    allocBuffers(tasks);

    std::pair<cv::Mat, cv::Mat> frames[BATCH_SIZE];
    std::pair<cv::Mat, cv::Mat> frame;

    if (!HEADLESS) {
        cv::namedWindow(WINDOW_NAME);
        cv::setWindowTitle(WINDOW_NAME, WINDOW_NAME);
    }

    std::cout << "Running...\n";

    auto cv_type = CV_8UC3;

    float* softmax = new float[400];
    float out_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[NUM_KERNELS-1], splitInfo[NUM_KERNELS-1].output_node.name.c_str()));
    float in_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[0], splitInfo[0].shifted_input_node.name.c_str()));

    printf("inscale: %f, outscale: %f\n", in_scale, out_scale);

    std::vector<std::string> kinetics_imgs;
    if (arg_path != "") {
        std::vector<std::string> temp_imgs = listDir(arg_path);
        float tick = temp_imgs.size() / 8.0;
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            int img_idx = tick / 2.0 + tick*i;
            kinetics_imgs.push_back(temp_imgs[img_idx]);
        }
	} else {
        std::vector<std::string> temp_imgs = listDir(KINETICS_PATH);
        float tick = temp_imgs.size() / 8.0;
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            int img_idx = tick / 2.0 + tick*i;
            kinetics_imgs.push_back(temp_imgs[img_idx]);
        }
	} 


    std::vector<cv::Mat> in_imgs;
    for (int i = 0; i < BATCH_SIZE; i++) {
        //in_imgs.emplace_back(224, 224, cv_type);
        in_imgs.emplace_back(
               splitInfo[0].shifted_input_node.shape[0],
               splitInfo[0].shifted_input_node.shape[1],
               cv_type);
               //dpuGetTensorAddress(dpuGetBoundaryIOTensor(tasks[0], splitInfo[0].shifted_input_node.name.c_str())) + i*splitInfo[0].shifted_input_node.batchSize());
               //splitInfo[0].shiftedAddr() + i*splitInfo[0].shifted_input_node.batchSize());
    }



    auto t_lastframe = high_resolution_clock::now();
    int frame_num = -1;
    for (;;) {
            frame_num++;
            t_lastframe = high_resolution_clock::now();

            auto t_preframe = high_resolution_clock::now();
            for (int i = 0; i < BATCH_SIZE; i++) {
                frames[i].first = cv::imread(kinetics_imgs[i]);
                    //frames[i].first = cv::Mat(224,224,cv_type,cvScalar(0));
                    cv::imshow(WINDOW_NAME, frames[i].first);
                    cv::waitKey(500);
            }

            auto t_postframe = high_resolution_clock::now();
            for (int segment = 0; segment < BATCH_SIZE; segment++) {
                cv::Mat frame = frames[segment].first.clone();
                cv::Mat frame_rgb;

                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
                frame.convertTo(frame_rgb, CV_32FC3, 1/255.0);
                int new_w = 256;
                int new_h = 256;
                if (frame.cols > frame.rows) {
                    new_w = 256*frame.cols/frame.rows;
                } else {
                    new_h = 256*frame.rows/frame.cols;
                }
                cv::resize(frame_rgb, frame_rgb, cv::Size(new_w, new_h));
                frame_rgb = frame_rgb(cv::Rect((new_w - 224)/2, (new_h - 224)/2, 224, 224)).clone();

                // Normalize:
                // mean = (0.485, 0.456, 0.406) = (124, i27, 104)
                // std = (0.229, 0.224, 0.225)
                //frame_rgb = (frame_rgb - cv::Scalar(124, 127, 104));
                frame_rgb -= cv::Scalar(0.485,0.456, 0.406);
                cv::divide(frame_rgb, cv::Scalar(0.229, 0.224, 0.225), frame_rgb);
                frame_rgb.convertTo(in_imgs[segment], cv_type, in_scale);
                int8_t* input_data = dpuGetTensorAddress(dpuGetBoundaryIOTensor(tasks[0], splitInfo[0].shifted_input_node.name.c_str())) + segment*splitInfo[0].shifted_input_node.batchSize();
                memcpy_assign(input_data, in_imgs[segment].data, splitInfo[0].shifted_input_node.batchSize());
            }
            auto t_postprocess = high_resolution_clock::now();

            runTSMSerial(tasks);
            int8_t* features = splitInfo[NUM_KERNELS-1].outAddr();

			float scaled_features[400] = {0.0f};
			for (int b = 0; b < BATCH_SIZE; b++) {
                for (int i = 0; i < 400; i++) {
                    scaled_features[i] += features[b*400 + i]*out_scale;
                }
			}

            for (int i = 0; i < 400; i++) {
                scaled_features[i] /= 8;
            }

            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>>
                top5;
            top5.push(std::make_pair(-1000, -1));
            top5.push(std::make_pair(-1000, -1));
            top5.push(std::make_pair(-1000, -1));
            top5.push(std::make_pair(-1000, -1));
            top5.push(std::make_pair(-1000, -1));

            for (int i = 0; i < 400; i++ ) {
                if (scaled_features[i] > top5.top().first) {
                    top5.pop();
                    top5.push(std::make_pair(scaled_features[i], i));
                }
            }
            std::vector<std::pair<float, int>> top5_list;
            while (!top5.empty()) {
                top5_list.push_back(top5.top());
                top5.pop();
            }
            printf("top5 = (%d, %f), (%d, %f), (%d, %f), (%d, %f), (%d, %f)\n",
                    top5_list[0].second, top5_list[0].first,
                    top5_list[1].second, top5_list[1].first,
                    top5_list[2].second, top5_list[2].first,
                    top5_list[3].second, top5_list[3].first,
                    top5_list[4].second, top5_list[4].first);

            printf("(1) air_drumming = %f\n", scaled_features[1]);
            auto t_postrun = high_resolution_clock::now();

            /*
            if (!HEADLESS) {
                cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
                cv::flip(frame, frame, 1);
                cv::imshow("TSM on FPGA", frame);
                int key = cv::waitKey(1);
                if (key == 0x71)
                    break;
            }
            */
            auto t_postshow = high_resolution_clock::now();

            std::uint64_t d_framegrab = std::chrono::duration_cast<std::chrono::microseconds>(t_postframe - t_preframe).count();
            std::uint64_t d_frameprocess = std::chrono::duration_cast<std::chrono::microseconds>(t_postprocess - t_postframe).count();
            std::uint64_t d_run = std::chrono::duration_cast<std::chrono::microseconds>(t_postrun - t_postprocess).count();
            std::uint64_t d_show = std::chrono::duration_cast<std::chrono::microseconds>(t_postshow - t_postrun).count();
            std::uint64_t d_total = std::chrono::duration_cast<std::chrono::microseconds>(t_postshow - t_preframe).count();
            printf("frameprocess: %lu; run: %lu; total: %lu\n", d_frameprocess, d_run, d_total);
    }

    delete softmax;
    freeBuffers();

    bool err = false;
    for (int i = 0; i < NUM_KERNELS; i++) {
        int t = dpuDestroyTask(tasks[i]);
        int d = dpuDestroyKernel(kernels[i]);
        if (d != 0 || t != 0)
            err = true;
    }
    if (err)
        fail("Error destroying kernel/task");


    if (dpuClose() != 0)
        fail("Error closing DPU device");

    return 0;
}

int main(int argc, char* argv[]) {

    if (argc > 1)
        arg_path = argv[1];

    cv::VideoCapture cap;
    if (!KINETICS_TEST) {
        char gst_pipeline[500];
        //sprintf(gst_pipeline, "filesrc location=%s ! qtdemux name=demux demux.video_0 ! h264parse ! video/x-h264, alignment=au ! omxh264dec low-latency=1 ! videoconvert ! videoscale ! video/x-raw ! queue max-size-bytes=0 ! kmssink bus-id=fd4a0000.zynqmp-display fullscreen-overlay=1",
        sprintf(gst_pipeline, "filesrc location=%s ! qtdemux name=demux demux.video_0 ! h264parse ! video/x-h264, alignment=au ! omxh264dec low-latency=1 ! videoconvert ! videoscale ! video/x-raw ! queue max-size-bytes=0 ! appsink",
                VID_PATH.c_str());
        cap.open(gst_pipeline, cv::CAP_GSTREAMER);

        if (!cap.isOpened())
            fail("Error opening camera");
    }

    int ret = run(cap);

    cv::destroyAllWindows();
    cap.release();

    return ret;
}
