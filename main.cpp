#include <iostream>
#include "vector"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

struct center{
    int l,a,b;
    int x,y;
};
center getmincenter(int x,int y);
double getdistance(center a,center b,double S);
void post_processing(Mat* image);

const double Emax = 0.01;
const int k = 100;//cluster num
vector<center> centers;
vector<vector<int>> label;
vector<vector<double>> dis;
vector<vector<int>> new_clusters;
Mat labimg, img;
int width, height;

int main() {
    img = imread("./lena.png");
    cvtColor(img,labimg,COLOR_BGR2Lab);
    namedWindow("origin img", WINDOW_NORMAL);
    imshow("origin img", img);

    width = img.cols;
    height = img.rows;
    const int N = width*height;
    const double S = sqrt(N/k);
    double E = 10;
    int iter = 1;
    cout<<"width: " << width <<" height: "<<height<<endl;
    /* init */
    for(int i = 0;i<height;i++)
    {
        vector<int> l;
        vector<double> d;
        for(int j = 0;j<width;j++)
        {
            l.push_back(-1);
            d.push_back(MAXFLOAT);
        }
        label.push_back(l);
        dis.push_back(d);
    }

    for(int i = S/2;i<height-S/2;i+=S)
    {
        for(int j = S/2;j<width-S/2;j+=S)
        {
            centers.push_back(getmincenter(j,i));
        }
    }

    /* main loop */
    while(E>Emax)
    {
        int center_count = 0;
        for(center c:centers){
            for(int i = c.x-S;i<=c.x+S;i++){
                for(int j = c.y-S;j<=c.y+S;j++){
                    if(i<0||i>=width||j<0||j>=height) continue;
                    center cur{labimg.at<Vec3b>(j,i)[0],labimg.at<Vec3b>(j,i)[1],labimg.at<Vec3b>(j,i)[2],i,j};
                    double d= getdistance(c,cur,S);
                    if(d<dis[j][i]){
                        dis[j][i] = d;
                        label[j][i] = center_count;
                    }
                }
            }
            center_count++;
        }
        /* update centers */
        vector<center> updated_centers = centers;
        vector<int> count;
        for(int i = 0;i<center_count;i++){
            updated_centers[i].l = 0;
            updated_centers[i].a = 0;
            updated_centers[i].b = 0;
            updated_centers[i].x = 0;
            updated_centers[i].y = 0;
            count.push_back(0);
        }
        for(int i = 0;i<width;i++){
            for(int j = 0;j<height;j++){
               int class_label = label[j][i];
               if(class_label!=-1){
                   updated_centers[class_label].l+=labimg.at<Vec3b>(j,i)[0];
                   updated_centers[class_label].a+=labimg.at<Vec3b>(j,i)[1];
                   updated_centers[class_label].b+=labimg.at<Vec3b>(j,i)[2];
                   updated_centers[class_label].x+=i;
                   updated_centers[class_label].y+=j;
                   count[class_label]++;
               }
            }
        }
        for(int i = 0;i<center_count;i++){
            updated_centers[i].l /= count[i];
            updated_centers[i].a /= count[i];
            updated_centers[i].b /= count[i];
            updated_centers[i].x /= count[i];
            updated_centers[i].y /= count[i];
        }
        /* update err */
        E=0;
        for(int i = 0;i<center_count;i++){
            E+=getdistance(centers[i],updated_centers[i],S);
        }
        centers = updated_centers;
        cout<< "iter:"<<iter++<<" err:" << E<<endl;
    }
    cout<<"the number of cluster center : "<< centers.size()<<endl;

    /* update img */
    for(int i = 0;i<width;i++){
        for(int j = 0;j<height;j++){
            int class_label = label[j][i];
            if(class_label!=-1){
                labimg.at<Vec3b>(j, i)[0] = centers[class_label].l;
                labimg.at<Vec3b>(j, i)[1] = centers[class_label].a;
                labimg.at<Vec3b>(j, i)[2] = centers[class_label].b;
            }
        }
    }
    post_processing(&labimg);
    cvtColor(labimg,img,COLOR_Lab2BGR);
    /* draw contours */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int num = 0;
            for (int ii = i - 1; ii <= i + 1 && num < 2; ii++) {
                for (int jj = j - 1; jj <= j + 1 && num < 2; jj++) {
                    if (ii < 0 || ii >= height || jj < 0 || jj >= width) continue;
                    if (new_clusters[ii][jj] != new_clusters[i][j]) num++;
                }
            }
            if(num >= 2){
                img.at<Vec3b>(i,j)[0] = 255;
                img.at<Vec3b>(i,j)[1] = 0;
                img.at<Vec3b>(i,j)[2] = 0;
            }
        }
    }
    //display_contours(&img);
    namedWindow("img", WINDOW_NORMAL);
    imshow("img", img);
    imwrite("lena_result.png",img);
    waitKey(0);
    return 0;
}

center getmincenter(int x,int y)
{
    int bestx,besty;
    int mingrad = 0xffffff;
    Vec3b best;
    for(int i = x-1;i<=x+1;i++)
    {
        for(int j = y-1;j<=y+1;j++)
        {
            int middle = labimg.at<Vec3b>(j,i)[0];
            int right =  labimg.at<Vec3b>(j,i+1)[0];
            int down =  labimg.at<Vec3b>(j+1,i)[0];

            if(sqrt(pow(right - middle,2)+pow(down - middle,2)) < mingrad) {
                mingrad = sqrt(pow(right - middle, 2) + pow(down - middle, 2));
                best = labimg.at<Vec3b>(j, i);
                bestx = i;
                besty = j;
            }
        }
    }
    //cout << (int)best[0]<<"  "<<(int)best[1]<<"  "<<(int)best[2]<<endl;
    return center{best[0],best[1],best[2],bestx,besty};
}

double getdistance(center a,center b,double S)
{
    double dc = sqrt(pow(a.l-b.l,2)+pow(a.a-b.a,2)+pow(a.b-b.b,2));
    double ds = sqrt(pow(a.x-b.x,2)+pow(a.y-b.y,2));
    return sqrt(dc*dc + pow(ds/S,2) * 100);
}

void post_processing(Mat* image)
{
    int label_count = 0, adjlabel = 0;
    const int lims = (width * height) / ((int)centers.size());

    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};

    /* Initialize the new cluster matrix. */

    for (int i = 0; i < height; i++) {
        vector<int> nc;
        for (int j = 0; j < width; j++) {
            nc.push_back(-1);
        }
        new_clusters.push_back(nc);
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (new_clusters[i][j] == -1) {
                vector<Point> elements;
                elements.push_back(Point(j, i));

                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];

                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        if (new_clusters[y][x] >= 0) {
                            adjlabel = new_clusters[y][x];
                        }
                    }
                }

                int count = 1;
                for (int c = 0; c < count; c++) {
                    for (int k = 0; k < 4; k++) {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];

                        if (x >= 0 && x < width && y >= 0 && y < height) {
                            if (new_clusters[y][x] == -1 && label[i][j] == label[y][x]) {
                                elements.push_back(Point(x, y));
                                new_clusters[y][x] = label_count;
                                count += 1;
                            }
                        }
                    }
                }

                /* Use the earlier found adjacent label if a segment size is
                   smaller than a limit. */
                if (count <= lims >> 2) {
                    for (int c = 0; c < count; c++) {
                        new_clusters[elements[c].y][elements[c].x] = adjlabel;
                    }
                    label_count -= 1;
                }
                label_count += 1;
            }
        }
    }


}

