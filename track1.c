/*
  lalala
  
  Perform single object tracking with particle filtering

  @author Rob Hess
  @version 1.0.0-20060306
*/
/************************************************************************/
/* 更多源码，欢迎访问http://www.cnblogs.com/yangyangcv                                                                     */
/************************************************************************/
#include "defs.h"
#include "utils.h"
#include "particles.h"
#include "observation.h"
using namespace std;

/********************************** Globals **********************************/
char* pname;                      /* program name */
char* vid_file = "test.avi";                   /* input video file name */
int num_particles = PARTICLES;    /* number of particles */
int show_all = 1;             /* TRUE to display all particles */
int export1 = FALSE;               /* TRUE to exported tracking sequence */
int degradation = 0;
struct timeval start; 
struct timeval end;
unsigned long nSecs = 0;
unsigned long total_time = 0;

/*********************************** Main ************************************/

int main( int argc, char** argv )
{
    gsl_rng* rng; //声明gsl随机数生成器
    IplImage* frame, * hsv_frame, * frames[MAX_FRAMES], *frmforp[6];
    IplImage** hsv_ref_imgs;  //hsv图
    histogram** ref_histos;  //直方图
    CvCapture* video;
    particle** particles, ** new_particles;
    CvScalar color;  //CvScalar就是一个可以用来存放4个double数值的数组,一般用来存放像素值
    CvRect* regions;
    CvRect* estimate;

    int num_objects = 0;
    int num_degradation = 0 ;
    int i, k, w, h, x, y, pf1, pf2, pf3, border,m;
    int j;

    /*******************随机数生成器**********************/
    gsl_rng_env_setup(); //读取环境变量GSL_RNG_TYPE和GSL_RNG_SEED的值，并把他们分别赋给gsl_rng_default和gsl_rng_default_seed
    rng = gsl_rng_alloc( gsl_rng_mt19937 );
    gsl_rng_set( rng, time(NULL) );

    char timetxtPathName[50];  //name of a text file to save time data of performing
    char estimatetxtPathName[50];  //name of a text file to save estimate results
    sprintf(timetxtPathName,"time.xsw");
    sprintf(estimatetxtPathName,"estimate,xsw");
    ofstream timetxt;
    ofstream estimatetxt;
    timetxt.open(timetxtPathName);
    estimatetxt.open(estimatetxtPathName);
    video = cvCaptureFromFile( vid_file ); //open video
    //video = cvCaptureFromCAM( 0);   // open camera
    if( ! video )
    {
        fatal_error("couldn't open video file %s", vid_file);
    }
    else
    {
        fprintf( stderr, "Video captured : %s\n", vid_file );
        fprintf( stderr, "FPS   : %.1f\n", cvGetCaptureProperty( video, CV_CAP_PROP_FPS ) );
        fprintf( stderr, "SIZE  : %.1f * %.1f\n", cvGetCaptureProperty( video, CV_CAP_PROP_FRAME_WIDTH ),
                 cvGetCaptureProperty( video, CV_CAP_PROP_FRAME_HEIGHT ) );
    }
    i = 0;
    while( frame = cvQueryFrame( video ) )
    {
        hsv_frame = bgr2hsv( frame );  //bgr to hsv
        frames[i] = (IplImage*)cvClone( frame );
        for(k = 0; k < 6 ; k++)
        {
            frmforp[k]= (IplImage*)cvClone( hsv_frame );
        }
        /* allow user to select object to be tracked in the first frame */
        if( i == 0 ) //第一帧
        {
            w = frame->width;
            h = frame->height;
            fprintf( stderr, "Select object region to track\n" );
            while( num_objects == 0 )
            {
                num_objects = get_regions( frame, &regions );  //get regions of targets
                /**********save the original regions' data***********/
                /*                timetxt<<regions[0].x<<" "<<regions[0].y<<" "<<regions[0].width<<" "<<regions[0].height<<endl;
                timetxt<<regions[1].x<<" "<<regions[1].y<<" "<<regions[1].width<<" "<<regions[1].height<<endl;
                timetxt<<regions[2].x<<" "<<regions[2].y<<" "<<regions[2].width<<" "<<regions[2].height<<endl;
                timetxt<<regions[3].x<<" "<<regions[3].y<<" "<<regions[3].width<<" "<<regions[3].height<<endl;*/
                regions[0] = cvRect(21,96,35,54); //cat.avi
                //regions[0] = cvRect(663,318,77,45); //tank5.avi
                //                regions[1] = cvRect(663,318,79,47);
                //                regions[2] = cvRect(349,187,69,73);
                //                regions[3] = cvRect(350,189,67,67);               
                if( num_objects == 0 )
                    fprintf( stderr, "Please select a object\n" );
            }

            /* compute reference histograms and distribute particles */
            ref_histos = compute_ref_histos( hsv_frame, regions, num_objects );  //计算直方图
            //if( export1 )
            //export_ref_histos( ref_histos, num_objects );  //是否输出直方图
            particles = init_distribution( regions, ref_histos, num_objects, num_particles ); //初始化分布粒子
            new_particles = ( particle** )malloc( num_objects * sizeof(particle*) );
            estimate = ( CvRect* )malloc( num_objects * sizeof(CvRect) );
            // particles = ( particle** )malloc( n * sizeof(particle*) );

            /***generate multi-random number generator*****/

            /* rng = (gsl_rng**)malloc(num_objects * sizeof(gsl_rng*));
            for(k = 0;k < num_objects;k++)
            {
                rng[k] = gsl_rng_alloc( gsl_rng_mt19937 );
                gsl_rng_set( rng[k], time(NULL) );
            }
            */
            border = num_objects*num_particles;

        }
        else //之后帧
        {
            gettimeofday(&start,NULL);
            color = CV_RGB(255,255,0);
            //     const int MIN_ITERATOR_NUM = 1; //假设单线程循环不小于4
            //     int ncore = omp_get_num_procs(); //获取执行核数量
            //     int max_tn= num_objects/MIN_ITERATOR_NUM;
            //     int tn = max_tn>2*ncore ? 2*ncore:max_tn; //核数量的两倍和最大需求线程数取较小值
            //     #pragma omp parallel for private(j)

            /* perform prediction and measurement for each particle */
            //       #pragma omp parallel //pf1为滤波器号，j为粒子号，frameN为各线程帧复制序号
            //       {
            //       #pragma omp for private(pf1,j)
            for( m = 0; m < border; m++ )
            {
                pf1 = m >> 7;  //除以粒子数128，用移位代替除法节约时间(移动7位)
                j = m % num_particles;
                //fprintf(stderr,"core %d \n",omp_get_num_procs());
                particles[pf1][j] = transition( particles[pf1][j], w, h, rng ); //paticles transition(w denotes weight,h
                //denotes height)
                //fprintf(stderr,"object %d threadnum %d particle %d\n",pf1,omp_get_thread_num(),j);
                //fprintf(stderr,"s%d = %f\n",pf1,particles[pf1][j].s);
                //#pragma omp critical //临界区，排队等候
                /* 并行用 particles[pf1][j].w = likelihood( frmforp[omp_get_thread_num()], cvRound(particles[pf1][j].y),
                                                     cvRound( particles[pf1][j].x ),
                                                     cvRound( particles[pf1][j].width * particles[pf1][j].s ),
                                                     cvRound( particles[pf1][j].height * particles[pf1][j].s ),
                                                     particles[pf1][j].histo );*/
                //fprintf(stderr,"%f\n",particles[pf1][j].w);
                /*likely = likelihood( frmforp[pf1], cvRound(particles[pf1][j].y),
                                                     cvRound( particles[pf1][j].x ),
                                                     cvRound( particles[pf1][j].width * particles[pf1][j].s ),
                                                     cvRound( particles[pf1][j].height * particles[pf1][j].s ),
                                                     particles[pf1][j].histo );*/
                particles[pf1][j].w *= likelihood( frmforp[pf1], cvRound(particles[pf1][j].y),
                                                   cvRound( particles[pf1][j].x ),
                                                   cvRound( particles[pf1][j].width * particles[pf1][j].s ),
                                                   cvRound( particles[pf1][j].height * particles[pf1][j].s ),
                                                   particles[pf1][j].histo );
                //fprintf(stderr,"%f\n",particles[pf1][j].w);
                if( show_all ) //display all particles
                    display_allparticle( frames[i], particles[pf1][j], color ); //画点
            }
            //     #pragma omp for
            for( pf2 = 0; pf2 < num_objects; pf2++)
            {
                /* normalize weights and resample a set of unweighted particles */
                normalize_weights( particles[pf2], num_particles );  //归一化权重
                degradation = degradation_estimate( particles[pf2], num_particles );
                qsort( particles[pf2], num_particles, sizeof( particle ), &particle_cmp );
                if(degradation <= 2*num_particles/3)
                {
                    //fprintf(stderr,"%d\n",1);
                    num_degradation ++;
                    //particles[pf2] = resample( particles[pf2], num_particles );
                    particles[pf2] = genetic(  particles[pf2], num_particles );
                    //new_particles[pf2] = resample( particles[pf2], num_particles );
                    //free( particles[pf2] );
                    //particles[pf2] = new_particles[pf2];
                }
                //free(new_particles[pf1]);
            }
            //              }
            gettimeofday(&end,NULL);
        }

        /* display most likely particle */
        color = CV_RGB(255,0,0);
        for( pf3= 0; pf3 < num_objects; pf3++)
        {
            result_estimate( particles[pf3], num_particles );
            //     fprintf(stderr,"%f %f %d %d\n",particles[pf3][0].x,particles[pf3][0].y,particles[pf3][0].width,particles[pf3][0].height);
            estimate[pf3] = display_bestparticle( frames[i], particles[pf3][num_particles-1], color );
            estimatetxt<<pf3+1<<" "<<cvRound(particles[pf3][num_particles-1].x)<<" "<<cvRound(particles[pf3][num_particles-1].y)<<" "
                       <<particles[pf3][num_particles-1].s<<endl;
            //     cvSaveImage("target.jpg",frames[i]);
        }
        cvNamedWindow("Video", 1 );
        cvMoveWindow("Video",0,0);
        cvShowImage( "Video", frames[i] );
        //cvSaveImage("target.jpg",frames[i]);
        // nSecs = end.tv_sec-start.tv_sec;
        // fprintf(stderr,"%d\n",end.tv_usec);
        //fprintf(stderr,"%d\n",nSecs);
        int key = cvWaitKey(100000);
        //fprintf(stderr,"%d\n",endpro);
        if( key == 1048603 || key == 27 )
            break;
        else if(key == 1048688 || key == 112)
        {
            fprintf(stderr,"frame:%d Pause! Press any key to continue ... \n",i+2);
            cvWaitKey(0);
        }
        cvReleaseImage( &hsv_frame );
        for(k = 0; k < 6 ; k++)
        {
            cvReleaseImage( &frmforp[k] );
        }
        i++;
        if(i > 600)
        { 
            break;
        }
        nSecs = 10000*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)/100;
        total_time += nSecs ;
        timetxt<<i<<" "<<nSecs<<endl;
    }
    estimatetxt<<"num of degradation is "<<num_degradation<<endl;
    timetxt<<"total time is "<<total_time<<"(.1ms)"<<endl;
    timetxt.close();
    estimatetxt.close();
    fprintf(stderr,"nframe is %d\n",i-1);
    cvReleaseCapture( &video );
    return 0 ;
}
