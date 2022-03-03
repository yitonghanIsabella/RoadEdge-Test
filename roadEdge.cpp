#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

int main()
{
    VideoCapture capture("C:/VisionImg/01.avi");
    if (!capture.isOpened())
        cout << "fail to open!" << endl;

    long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
    //cout << "totalFrameNumber" << totalFrameNumber << endl;
    long frameToStart = 0;
    capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
    int frameToStop = totalFrameNumber;
    double rate = capture.get(CV_CAP_PROP_FPS);

    bool stop = false;
    Mat frame, basic, dst;
    //两帧间的间隔时间:
    int delay = 25 / rate;
    //int delay = 10000 / rate;
    long currentFrame = frameToStart;

    //滤波器的核
    int kernel_size = 3;
    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);

    float gap1 = 0, gap2 = 0;
    vector<float> start(2, 0);
    vector<float> end(2, 0);
    vector<float> threeK(3, 0);
    threeK[0] = 1.7, threeK[1] = 1.9, threeK[2] = 2.15;
    vector<float> edge(4, 0);

    VideoWriter writer;
    writer.open("C:/VisionImg/04.avi", -1,
        rate,   //录制时的帧率
        Size(frame.cols, frame.rows),
        true);
    if (!writer.isOpened())
    {
        cout << "存储视频失败" << endl;
    }

    while (!stop)
    {
        //cout << "第" << currentFrame << "帧：" << endl;
        if (!capture.read(frame))
        {
            cout << "读取视频失败" << endl;
            return -1;
        }
        filter2D(frame, basic, -1, kernel);
        Canny(basic, dst, 50, 200, 3);

        size_t i = 0, j = 0;
        vector<Vec4i> edges;
        vector<float> edgesK;
        vector<Vec4i> lines;
        HoughLinesP(dst, lines, 1, CV_PI / 180, 200, 30, 30);
        vector<int> flag(lines.size(), 0);
        vector<float> k(lines.size(), 0);
        for (i = 0; i < lines.size(); i++)
        {
            Vec4i l = lines[i];
            k[i] = (((l[3] - l[1]) + 0.0) / (l[2] - l[0]));
            float b = (l[0] * l[3] - l[2] * l[1] + 0.0) / (l[0] - l[2]);

            if (k[i] > 1.4 && k[i] < 2.5)
            {
                for (j = 0; j < i; j++) {
                    if (flag[j] == 1 && fabs(k[i] - k[j]) < 0.15) {
                        break;
                    }
                }
                if (j == i) {
                    //cout << "k[" << i << "]=" << k[i] << endl;
                    flag[i] = 1;

                    l[0] = -1.0 * b / k[i];
                    l[1] = 0;
                    l[3] = 479;
                    l[2] = (l[3] - b + 0.0) / k[i];

                    edges.push_back(l);
                    edgesK.push_back(k[i]);
                    //line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);
                }
            }
        }

        //cout << "灭点:" << end[0] << "," << end[1] << endl;

        if (edges.size() > 1) {
            float k1 = edgesK[0], k2 = edgesK[1];
            int x1 = edges[0][0], y1 = edges[0][1];
            int x2 = edges[1][0], y2 = edges[1][1];
            end[0] = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2);
            end[1] = (-k2 * x2 * k1 + y2 * k1 + k1 * k2 * x1 - y1 * k2) / (k1 - k2);
        }

        if (edges.size() == 3)
        {
            int max = 0, min = 0;    //最大和最小斜率直线的下标
            for (i = 0; i < edgesK.size(); i++) {
                if (edgesK[max] < edgesK[i])
                    max = i;
                if (edgesK[min] > edgesK[i])
                    min = i;
            }
            int mid = 3 - max - min;

            threeK[0] = edgesK[min];           //更新
            threeK[1] = edgesK[mid];
            threeK[2] = edgesK[max];
            gap1 = edgesK[max] - edgesK[mid];
            gap2 = edgesK[max] - edgesK[min];
            edge[0] = edges[max][0];
            edge[1] = edges[max][1];
            edge[2] = edges[max][2];
            edge[3] = edges[max][3];

            start[0] = edges[max][2];           //路沿线起点
            start[1] = edges[max][3];
            line(frame, Point(edges[max][0], edges[max][1]), Point(edges[max][2], edges[max][3]), Scalar(0, 0, 255), 3, LINE_AA);
        }
        else if (edges.size() < 1) {
            start[0] = edge[2];
            start[1] = edge[3];
            line(frame, Point(edge[0], edge[1]), Point(edge[2], edge[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }
        else
        {
            //cout<<"3个k:"<< threeK[0] <<" " << threeK[1] <<" " << threeK[2] << endl;
            //cout << "2个gap:" << gap1 << " " << gap2 << endl;

            float kMax = 1, max_index = 0, another_index = 0;
            int flag2 = 0, flag1 = 0, flag0 = 0, flag3 = 0;

            if (edges.size() == 2) {
                if (edgesK[0] < edgesK[1]) {
                    swap(edgesK[0], edgesK[1]);
                    swap(edges[0], edges[1]);
                }
            }
            if (threeK[2] < 2.0 || threeK[0] < 1.5) {
                threeK[2] = edgesK[0];
                threeK[1] = threeK[2] - gap1;
                threeK[0] = threeK[2] - gap2;
                flag3 = 1;
            }

            for (i = 0; i < edges.size(); i++) {
                if (fabs(edgesK[i] - threeK[2]) < fabs(edgesK[i] - threeK[1])) {
                    edge[0] = edges[i][0];  //更新
                    edge[1] = edges[i][1];
                    edge[2] = edges[i][2];
                    edge[3] = edges[i][3];
                    threeK[2] = edgesK[i];

                    max_index = i;
                    flag2 = 1;
                    start[0] = edges[i][2];
                    start[1] = edges[i][3];
                    line(frame, Point(edges[i][0], edges[i][1]), Point(edges[i][2], edges[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
                }
                else if (fabs(edgesK[i] - threeK[1]) < fabs(edgesK[i] - threeK[0])) {
                    if (!flag3)
                        threeK[1] = edgesK[i];
                    another_index = i;
                    kMax = edgesK[i] + gap1;
                    flag1 = 1;
                }
                else {
                    if (!flag3)
                        threeK[0] = edgesK[i];
                    another_index = i;
                    kMax = edgesK[i] + gap2;
                    flag0 = 1;
                }
            }
            //cout << "flags:" << flag0 << " " << flag1 << " " << flag2 << endl;
            //检测到路沿
            if (flag2) {
                if (flag1)
                    gap1 = edgesK[max_index] - edgesK[another_index];
                if (flag0)
                    gap2 = edgesK[max_index] - edgesK[another_index];
            }
            //没检测到路沿
            else {
                if (flag1 || flag0) {
                    start[1] = 479;
                    start[0] = (kMax * end[0] - end[1] + start[1]) / kMax;

                    edge[0] = end[0];   //更新
                    edge[1] = end[1];
                    edge[2] = start[0];
                    edge[3] = start[1];

                    line(frame, Point(end[0], end[1]), Point(start[0], start[1]), Scalar(0, 0, 255), 3, LINE_AA);
                }
            }
        }

        Point Start(start[0], start[1]);
        circle(frame, Start, 6, Scalar(0, 255, 255), -1);

        Point End(end[0], end[1]);
        //circle(frame, End, 3, Scalar(255, 0, 0), -1);      
        cout << endl;

        //writer << frame;
        imshow("result", frame);
        waitKey(delay);
        if (currentFrame > frameToStop)
            stop = true;
        currentFrame++;
    }
    writer.release();
    capture.release();
    waitKey(0);
    return 0;
}