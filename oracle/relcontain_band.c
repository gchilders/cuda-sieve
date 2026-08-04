/* Relation containment, inverting through OUR basis (read from bench's log),
 * so "outside our sieve region" is separated from "inside but not a survivor". */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef struct { long long q,rho,a0,a1,b0,b1; long I,J; uint8_t*s0,*s1;
                 long rel,in,out,miss; int same; } L_t;
static long gg(long a,long b){while(b){long t=a%b;a=b;b=t;}return a;}
static int bit(const uint8_t*m,long x){return (m[x>>3]>>(x&7))&1;}
static long long mulmod(long long a,long long b,long long m){return (long long)((__int128)a*b%m);} 
int main(int argc,char**argv){
    FILE*L=fopen(argv[1],"r"); static L_t Q[4096]; int nq=0; char line[16384];
    while(fgets(line,sizeof line,L)){
        if(strncmp(line,"# Sieving side-",15)) continue;
        char*p=strstr(line,"q="); if(!p) continue;
        L_t N; memset(&N,0,sizeof N); long long la0,lb0,la1,lb1;
        if(sscanf(p,"q=%lld; rho=%lld; a0=%lld; b0=%lld; a1=%lld; b1=%lld; I=%ld; J=%ld",
                  &N.q,&N.rho,&la0,&lb0,&la1,&lb1,&N.I,&N.J)!=8) continue;
        int dup=0; for(int k=0;k<nq;k++) if(Q[k].q==N.q&&Q[k].rho==N.rho){dup=1;break;}
        if(dup) continue;
        /* our basis, from bench's own log */
        char f[512]; snprintf(f,sizeof f,"bm_%lld_%lld_s1.log",N.q,N.rho);
        FILE*g=fopen(f,"r"); if(!g) continue; char gl[4096]; int got=0;
        while(fgets(gl,sizeof gl,g)){ char*t=strstr(gl,"basis (a0,a1,b0,b1) = (");
            if(!t) continue;
            if(sscanf(t+23,"%lld,%lld,%lld,%lld",&N.a0,&N.a1,&N.b0,&N.b1)==4) got=1; break; }
        fclose(g); if(!got) continue;
        /* does our basis equal las's up to negating the first vector? */
        N.same = ((N.a0==la0&&N.a1==lb0)||(N.a0==-la0&&N.a1==-lb0)) &&
                 ((N.b0==la1&&N.b1==lb1)||(N.b0==-la1&&N.b1==-lb1));
        size_t nb=(size_t)(N.I*N.J)/8;
        for(int s=0;s<2;s++){ snprintf(f,sizeof f,"%s/surv.q%lld.r%lld.side%d.bits",argv[2],N.q,N.rho,s);
            FILE*h=fopen(f,"rb"); if(!h) continue; uint8_t*m=malloc(nb);
            if(fread(m,1,nb,h)!=nb){free(m);fclose(h);continue;} fclose(h);
            if(s) N.s1=m; else N.s0=m; }
        if(N.s0&&N.s1&&nq<4096) Q[nq++]=N;
    }
    rewind(L);
    long trel=0,tin=0,tout=0,tmiss=0,tnoq=0; int nsame=0,ndiff=0;
    for(int k=0;k<nq;k++){ if(Q[k].same) nsame++; else ndiff++; }
    while(fgets(line,sizeof line,L)){
        if(line[0]!='-'&&(line[0]<'0'||line[0]>'9')) continue;
        long long a,b; if(sscanf(line,"%lld,%lld:",&a,&b)!=2) continue;
        trel++; int h=-1;
        for(int k=0;k<nq;k++){ long long r=((a-mulmod(Q[k].rho,b,Q[k].q))%Q[k].q+Q[k].q)%Q[k].q; if(!r){h=k;break;} }
        if(h<0){tnoq++;continue;}
        L_t*q=&Q[h]; q->rel++;
        long long det=q->a0*q->b1-q->a1*q->b0;
        int inreg=0,found=0;
        for(int neg=0;neg<2&&!found;neg++){
            long long aa=neg?-a:a, bb=neg?-b:b;
            long long ni=aa*q->b1-bb*q->b0, nj=q->a0*bb-q->a1*aa;
            if(!det||ni%det||nj%det) continue;
            long i=(long)(ni/det), j=(long)(nj/det);
            if(j<0||j>=q->J||i< -q->I/2||i>=q->I/2) continue;
            inreg=1;
            long x=(long)j*q->I+i+q->I/2;
            if(bit(q->s1,x)&&bit(q->s0,x)&&gg(i<0?-i:i,j)==1) found=1;
        }
        if(found){tin++;q->in++;} else if(!inreg){tout++;q->out++;} else {tmiss++;q->miss++;}
    }
    printf("lattices: %d  (basis matches las: %d, differs: %d)\n",nq,nsame,ndiff);
    printf("relations: %ld   unattributed: %ld\n",trel,tnoq);
    printf("  CONTAINED (in our region, is a survivor): %ld\n",tin);
    printf("  outside our sieve region (basis differs): %ld\n",tout);
    printf("  in region but NOT a survivor  <-- real : %ld\n",tmiss);
    long sr=0,si=0,dr=0,di=0,dout=0;
    for(int k=0;k<nq;k++){ if(Q[k].same){sr+=Q[k].rel;si+=Q[k].in;} else {dr+=Q[k].rel;di+=Q[k].in;dout+=Q[k].out;} }
    printf("\n  basis-matching lattices: %ld relations, %ld contained (%.3f%%)\n",sr,si,sr?100.0*si/sr:0);
    printf("  basis-differing lattices: %ld relations, %ld contained, %ld outside region\n",dr,di,dout);
    return 0;
}
