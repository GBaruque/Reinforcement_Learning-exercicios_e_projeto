#include <math.h>
#include <string.h>
#include <time.h>

#include <mex.h>
#include "memstring.h"
#include "pendtd.h"

#ifndef M_PI
  #define M_PI 3.1415927
#endif
#define INDEX(s, a) ((s)+(a)*observations_*observations_)

static Learner g_learner;

int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
}

float mymin(float a, float b)
{
  if (a < b) return a;
  else       return b;
}

float mymax(float a, float b)
{
  if (a > b) return a;
  else       return b;
}

mxArray *Learner::get()
{
  const char *fieldnames[] = {"alpha", "gamma", "epsilon", "lambda",
                              "observations", "actions", 
                              "goal_weight", "quadratic_weight", "action_weight", "time_weight",
                              "on_policy", "report_tests", "episodes", "initial", "tau", "step", "steps"};

  mxArray *pm = mxCreateStructMatrix(1, 1, 17, fieldnames);

  std::string str;

  mxSetField(pm, 0, "alpha", mxCreateDoubleScalar(alpha_));  
  mxSetField(pm, 0, "gamma", mxCreateDoubleScalar(gamma_));  
  mxSetField(pm, 0, "epsilon", mxCreateDoubleScalar(epsilon_));  
  mxSetField(pm, 0, "lambda", mxCreateDoubleScalar(lambda_));  
  mxSetField(pm, 0, "observations", mxCreateDoubleScalar(observations_));
  mxSetField(pm, 0, "actions", mxCreateDoubleScalar(actions_));  
  mxSetField(pm, 0, "goal_weight", mxCreateDoubleScalar(goal_weight_));  
  mxSetField(pm, 0, "quadratic_weight", mxCreateDoubleScalar(quadratic_weight_));  
  mxSetField(pm, 0, "action_weight", mxCreateDoubleScalar(action_weight_));  
  mxSetField(pm, 0, "time_weight", mxCreateDoubleScalar(time_weight_));  
  mxSetField(pm, 0, "on_policy", mxCreateDoubleScalar(on_policy_));  
  mxSetField(pm, 0, "report_tests", mxCreateDoubleScalar(report_tests_));  
  mxSetField(pm, 0, "episodes", mxCreateDoubleScalar(episodes_));  
  mxSetField(pm, 0, "initial", mxCreateDoubleScalar(initial_));
  mxSetField(pm, 0, "tau", mxCreateDoubleScalar(tau_));
  mxSetField(pm, 0, "step", mxCreateDoubleScalar(step_));
  mxSetField(pm, 0, "steps", mxCreateDoubleScalar(steps_));

  return pm;
}

void Learner::set(const mxArray *pm)
{
  mxArray *f;

  if ((f = mxGetField(pm, 0, "alpha"))) alpha_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "gamma"))) gamma_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "epsilon"))) epsilon_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "lambda"))) lambda_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "observations"))) observations_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "actions"))) actions_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "goal_weight"))) goal_weight_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "quadratic_weight"))) quadratic_weight_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "action_weight"))) action_weight_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "time_weight"))) time_weight_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "on_policy"))) on_policy_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "report_tests"))) report_tests_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "episodes"))) episodes_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "initial"))) initial_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "tau"))) tau_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "step"))) step_ = mxGetScalar(f);
  if ((f = mxGetField(pm, 0, "steps"))) steps_ = mxGetScalar(f);

  if (q_) free(q_);
  q_ = NULL;
  
  if (curve_) free(curve_);
  curve_ = NULL;
}

mxArray *Learner::curve()
{
  mxArray *curve = mxCreateDoubleMatrix(3, episodes_, mxREAL);

  if (curve_)
  {
    double *dbl = mxGetPr(curve);
    for (size_t ii=0; ii < 3*episodes_; ++ii)
      dbl[ii] = curve_[ii];
  }    

  return curve;
}

mxArray *Learner::path()
{
  mxArray *path = mxCreateDoubleMatrix(3, steps_, mxREAL);

  if (path_)
  {
    double *dbl = mxGetPr(path);
    for (size_t ii=0; ii < 3*steps_; ++ii)
      dbl[ii] = path_[ii];
  }    

  return path;
}

mxArray *Learner::q()
{
  mwSize dims[3] = {observations_, observations_, actions_};
  mxArray *q = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);

  if (q_)
  {
    double *dbl = mxGetPr(q);
    for (size_t ii=0; ii < observations_*observations_*actions_; ++ii)
      dbl[ii] = q_[ii];
  }

  return q;
}

void Learner::run()
{
  if (q_) free(q_);
  q_ = (float*) malloc(observations_*observations_*actions_*sizeof(float));

  for (size_t ii=0; ii < observations_*observations_*actions_; ++ii)
    q_[ii] = initial_ + 0.0001*(double(rand())/RAND_MAX);

  if (curve_) free(curve_);
  curve_ = (float*) malloc(3*episodes_*sizeof(float));
  
  for (size_t ii=0; ii < 3*episodes_; ++ii)
    curve_[ii] = 0;

  if (path_) free(path_);
  path_ = (float*) malloc(3*steps_*sizeof(float));
  
  for (size_t ii=0; ii < 3*steps_; ++ii)
    path_[ii] = 0;

  int *e = new int[steps_];
  for (size_t ii=0; ii < steps_; ++ii)
    e[ii] = -1;
    
  for (int ee=0; ee < episodes_; ++ee)
  {
    State s  = State(M_PI, 0);
    int   si = discretize(s); 
    int   ai = explore(act(si));
    float cr = 0;
    
    for (int ss=0; ss < steps_; ++ss)
    {
      float r, delta;
      State sp = step(s, ai, &r);
      int   spi = discretize(sp);
      int   api = act(spi);
      float u = (ai/(actions_-1.)-0.5)*2*tau_;
            
      path_[3*ss+0] = s.pos;
      path_[3*ss+1] = s.vel;
      path_[3*ss+2] = u;

      if (on_policy_)
      {
        api = explore(api);
        delta = r + gamma_*q_[INDEX(spi,api)] - q_[INDEX(si,ai)];
      }
      else
      {
        delta = r + gamma_*q_[INDEX(spi,api)] - q_[INDEX(si,ai)];
        api = explore(api);
      }

      e[ss] = INDEX(si,ai);
      float d = 1.0;
      for (int tt=ss; tt >= 0 && d > 0.0005; --tt)
      {
        if (tt!=ss && e[tt] == e[ss])
          e[tt] = -1;
        else if (e[tt] != -1)
          q_[e[tt]] += alpha_*delta*d;

        d *= gamma_*lambda_;
      }
      
      curve_[3*ee+0] += r;
      curve_[3*ee+1] += sp.pos*sp.pos;
      curve_[3*ee+2] += u*u;

      s = sp;
      si = spi;
      ai = api;
    }
    
    if (report_tests_)
    {
      curve_[3*ee+0] = curve_[3*ee+0] = curve_[3*ee+0] = 0;
      s = State(M_PI, 0);

      for (int ss=0; ss < steps_; ++ss)
      {
        float r;
        int ai = act(discretize(s));
        float u = (ai/(actions_-1.)-0.5)*2*tau_;
        
        path_[3*ss+0] = s.pos;
        path_[3*ss+1] = s.vel;
        path_[3*ss+2] = u;
        
        s = step(s, ai, &r);

        curve_[3*ee+0] += r;
        curve_[3*ee+1] += s.pos*s.pos;
        curve_[3*ee+2] += u*u;
      }
    }
  }
  
  delete[] e;
}

int Learner::discretize(State s)
{
  int pi = round_int((s.pos+M_PI)/(2*M_PI)*observations_);
  if (pi == observations_) pi = 0;

  int vi = round_int((mymax(mymin(s.vel, 12*M_PI), -12*M_PI)+12*M_PI)/(24*M_PI)*(observations_-1));
  
  return vi+(pi*observations_);
}

int Learner::act(int si)
{
  int mai=0;
  
  for (int ii=1; ii < actions_; ++ii)
    if (q_[INDEX(si,ii)] > q_[INDEX(si,mai)])
      mai = ii;
  
  return mai;
}

int Learner::explore(int ai)
{
  if ((double(rand())/RAND_MAX) < epsilon_)
    return rand()%actions_;
  else
    return ai;
}

State Learner::step(State s, int ai, float *r)
{
  static double J = 0.000191, m = 0.055, g = 9.81, l = 0.042, b = 0.000003, K = 0.0536, R = 9.5;

  double a   = s.pos;
  double ad  = s.vel;
  double u   = (ai/(actions_-1.)-0.5)*2*tau_;
  double add = (1/J)*(m*g*l*sin(a)-b*ad-(K*K/R)*ad+(K/R)*u);
  
  State sp(a+step_*ad, ad+step_*add);
  
  if (sp.pos > M_PI) sp.pos -= 2*M_PI;
  if (sp.pos < -M_PI) sp.pos += 2*M_PI;

  *r = goal_weight_      * (fabs(sp.pos) < 0.05*M_PI && fabs(sp.vel) < 0.5*M_PI) + 
       quadratic_weight_ * (5*sp.pos*sp.pos + 0.1*sp.vel*sp.vel) +
       action_weight_    * (1*u*u) + 
       time_weight_      * (1);

  // Normalize for step size       
  *r = *r * (step_/0.03);

  return sp;
}

void mexFunction(int nlhs, mxArray *plhs[ ],
                 int nrhs, const mxArray *prhs[ ])
{
  MexMemString func;

  if (nrhs < 1 || !mxIsChar(prhs[0]) || !(func = mxArrayToString(prhs[0])))
    mexErrMsgTxt("Missing function name.");

  if (!strcmp(func, "get"))
  {
    plhs[0] = g_learner.get();
  }
  else if (!strcmp(func, "set"))
  {
    if (nrhs < 2 || !mxIsStruct(prhs[1]))
      mexErrMsgTxt("Missing options.");

    g_learner.set(prhs[1]);
  }
  else if (!strcmp(func, "run"))
  {
    g_learner.run();

    plhs[0] = g_learner.curve();
    if (nlhs > 1)
      plhs[1] = g_learner.q();
    if (nlhs > 2)
      plhs[2] = g_learner.path();
  }
  else
    mexErrMsgTxt("Unknown command.");
}
