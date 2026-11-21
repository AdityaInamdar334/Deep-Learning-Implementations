#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { size_t rows, cols; double *data; } Mat;
#define AT(m,r,c) ((m).data[(r)*(m).cols+(c)])
static Mat mat(size_t r,size_t c) { Mat m={r,c,calloc(r*c,sizeof(double))}; assert(m.data); return m; }
static void drop(Mat *m){free(m->data);m->data=NULL;}
static void fill(Mat m,double val){for(size_t i=0;i<m.rows*m.cols;i++)m.data[i]=val;}
static void copy_top_left(Mat dst,Mat src){assert(dst.rows>=src.rows&&dst.cols>=src.cols);for(size_t r=0;r<src.rows;r++)for(size_t c=0;c<src.cols;c++)AT(dst,r,c)=AT(src,r,c);}
static void mul(Mat out,Mat a,Mat b){assert(a.cols==b.rows&&out.rows==a.rows&&out.cols==b.cols);for(size_t i=0;i<out.rows;i++)for(size_t j=0;j<out.cols;j++){double v=0;for(size_t k=0;k<a.cols;k++)v+=AT(a,i,k)*AT(b,k,j);AT(out,i,j)=v;}}
static double sigmoid(double x){return x>=0?1.0/(1.0+exp(-x)):exp(x)/(1.0+exp(x));}
static void activate(Mat dst,Mat src){assert(dst.rows>=src.rows&&dst.cols>=src.cols);for(size_t r=0;r<src.rows;r++)for(size_t c=0;c<src.cols;c++)AT(dst,r,c)=sigmoid(AT(src,r,c));}
static uint64_t rng_state=42;
static double rnd(void){rng_state^=rng_state>>12;rng_state^=rng_state<<25;rng_state^=rng_state>>27;return ((rng_state*UINT64_C(2685821657736338717))>>11)*(1.0/9007199254740992.0);}

enum{N=4,IN=2,H=4,OUT=1};
typedef struct{Mat X,Y,A0,Z0,A1,Z1,P,W0,W1,G0,G1;}Net;
static Net create(void){Net n={0};n.X=mat(N,IN);n.Y=mat(N,OUT);n.A0=mat(N,IN+1);n.Z0=mat(N,H);n.A1=mat(N,H+1);n.Z1=mat(N,OUT);n.P=mat(N,OUT);n.W0=mat(IN+1,H);n.W1=mat(H+1,OUT);n.G0=mat(IN+1,H);n.G1=mat(H+1,OUT);double x[8]={0,0,0,1,1,0,1,1};double y[4]={0,1,1,0};memcpy(n.X.data,x,sizeof x);memcpy(n.Y.data,y,sizeof y);fill(n.A0,1);fill(n.A1,1);copy_top_left(n.A0,n.X);for(size_t i=0;i<n.W0.rows*n.W0.cols;i++)n.W0.data[i]=(rnd()-.5)*3;for(size_t i=0;i<n.W1.rows*n.W1.cols;i++)n.W1.data[i]=(rnd()-.5)*3;return n;}
static void destroy(Net *n){Mat* mats[]={&n->X,&n->Y,&n->A0,&n->Z0,&n->A1,&n->Z1,&n->P,&n->W0,&n->W1,&n->G0,&n->G1};for(size_t i=0;i<sizeof mats/sizeof mats[0];i++)drop(mats[i]);}
static double forward(Net*n){mul(n->Z0,n->A0,n->W0);activate(n->A1,n->Z0);mul(n->Z1,n->A1,n->W1);activate(n->P,n->Z1);double loss=0;for(size_t i=0;i<N;i++){double d=AT(n->P,i,0)-AT(n->Y,i,0);loss+=.5*d*d;}return loss/N;}
static void numerical(Net*n,double epsilon){Mat *weights[]={&n->W0,&n->W1},*grads[]={&n->G0,&n->G1};for(int k=0;k<2;k++){Mat w=*weights[k],g=*grads[k];for(size_t i=0;i<w.rows*w.cols;i++){double original=w.data[i];w.data[i]=original+epsilon;double plus=forward(n);w.data[i]=original-epsilon;double minus=forward(n);w.data[i]=original;g.data[i]=(plus-minus)/(2*epsilon);}}}
static void analytical(Net*n){(void)forward(n);fill(n->G0,0);fill(n->G1,0);for(size_t i=0;i<N;i++){double p=AT(n->P,i,0),y=AT(n->Y,i,0);double delta=(p-y)*p*(1-p)/N;for(size_t j=0;j<H+1;j++)AT(n->G1,j,0)+=AT(n->A1,i,j)*delta;for(size_t j=0;j<H;j++){double h=AT(n->A1,i,j);double dh=delta*AT(n->W1,j,0)*h*(1-h);for(size_t k=0;k<IN+1;k++)AT(n->G0,k,j)+=AT(n->A0,i,k)*dh;}}}
static void step(Mat w,Mat grad,double lr){for(size_t i=0;i<w.rows*w.cols;i++)w.data[i]-=lr*grad.data[i];}
static void gradient_check(Net*n){analytical(n);size_t count=n->G0.rows*n->G0.cols+n->G1.rows*n->G1.cols;double* saved=malloc(count*sizeof(double));assert(saved);size_t z=n->G0.rows*n->G0.cols;memcpy(saved,n->G0.data,z*sizeof(double));memcpy(saved+z,n->G1.data,(count-z)*sizeof(double));numerical(n,1e-5);double maxerr=0;for(size_t i=0;i<z;i++){double d=fabs(saved[i]-n->G0.data[i]);if(d>maxerr)maxerr=d;}for(size_t i=0;i<count-z;i++){double d=fabs(saved[z+i]-n->G1.data[i]);if(d>maxerr)maxerr=d;}printf("gradient check: max absolute error %.3g\n",maxerr);assert(maxerr<1e-6);free(saved);}
static void test(void){Mat a=mat(2,2),b=mat(2,2),o=mat(2,2);AT(a,0,0)=1;AT(a,0,1)=2;AT(a,1,0)=3;AT(a,1,1)=4;AT(b,0,0)=5;AT(b,0,1)=6;AT(b,1,0)=7;AT(b,1,1)=8;mul(o,a,b);assert(AT(o,0,0)==19&&AT(o,0,1)==22&&AT(o,1,0)==43&&AT(o,1,1)==50);drop(&a);drop(&b);drop(&o);Mat d=mat(2,3),s=mat(2,2);fill(d,1);AT(s,0,0)=0;AT(s,0,1)=2;AT(s,1,0)=3;AT(s,1,1)=4;copy_top_left(d,s);assert(AT(d,0,2)==1&&AT(d,1,2)==1&&AT(d,1,0)==3);drop(&d);drop(&s);Net n=create();gradient_check(&n);destroy(&n);puts("tests passed");}
int main(int argc,char**argv){if(argc>1&&!strcmp(argv[1],"test")){test();return 0;}const int finite=argc>1&&!strcmp(argv[1],"finite");const int epochs=finite?25000:20000;const double lr=finite?2.0:2.0;Net n=create();gradient_check(&n);for(int i=0;i<epochs;i++){double loss=forward(&n);if(i%2500==0)printf("epoch %5d loss %.8f\n",i,loss);if(finite)numerical(&n,1e-4);else analytical(&n);step(n.W0,n.G0,lr);step(n.W1,n.G1,lr);}printf("final loss %.8f\n",forward(&n));int correct=0;for(size_t i=0;i<N;i++){double p=AT(n.P,i,0);int label=p>=.5;int expected=(int)AT(n.Y,i,0);correct+=label==expected;printf("%g XOR %g -> %.5f (class %d, target %d)\n",AT(n.X,i,0),AT(n.X,i,1),p,label,expected);}destroy(&n);if(correct!=N){fprintf(stderr,"not all XOR cases learned\n");return 1;}return 0;}
