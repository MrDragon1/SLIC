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

const double Emax = 0.01;
const int k = 500;//cluster num
vector<center> centers;
vector<vector<int>> label;
vector<vector<double>> dis;
Mat labimg, img;


int main() {
    img = imread("./lena.png");
    cvtColor(img,labimg,COLOR_BGR2Lab);
    namedWindow("origin img", WINDOW_NORMAL);
    imshow("origin img", img);

    const int width = img.cols;
    const int height = img.rows;
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
                bool flag = true;
                for(int ii = i-1;ii<=i+1&&flag;ii++){
                    for(int jj = j-1;jj<=j+1&&flag;jj++){
                        if(ii<0||ii>=width||jj<0||jj>=height) continue;
                        if(label[jj][ii]!=class_label) flag = false;
                    }
                }
                if(1){
                    labimg.at<Vec3b>(j, i)[0] = centers[class_label].l;
                    labimg.at<Vec3b>(j, i)[1] = centers[class_label].a;
                    labimg.at<Vec3b>(j, i)[2] = centers[class_label].b;
                }
                else{
                    labimg.at<Vec3b>(j, i)[0] = 0;
                    labimg.at<Vec3b>(j, i)[1] = 0;
                    labimg.at<Vec3b>(j, i)[2] = 0;
                }
            }
        }
    }
    cvtColor(labimg,img,COLOR_Lab2BGR);
    namedWindow("img", WINDOW_NORMAL);
    imshow("img", img);

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