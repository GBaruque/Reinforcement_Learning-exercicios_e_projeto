#include <mex.h>

struct State
{
  float pos, vel;
  
  State(float _pos, float _vel) : pos(_pos), vel(_vel) { }
};


class Learner
{
  protected:
    float alpha_, gamma_, epsilon_, lambda_;
    int observations_, actions_, episodes_, steps_;
    float goal_weight_, quadratic_weight_, action_weight_, time_weight_;
    bool on_policy_, report_tests_;
    float initial_, tau_, step_;
    
    float *q_, *curve_, *path_;

  public:
    Learner() :
      alpha_(0.2), gamma_(0.97), epsilon_(0.05), lambda_(0.67),
      observations_(40), actions_(3), episodes_(1000), steps_(100),
      goal_weight_(0), quadratic_weight_(-1), action_weight_(-1), time_weight_(0),
      on_policy_(true), report_tests_(true),
      initial_(0), tau_(3), step_(0.03), q_(NULL), curve_(NULL), path_(NULL)
    {
      srand(time(NULL));
    }
    
    ~Learner()
    {
      if (q_) free(q_);
      if (curve_) free(curve_);
      if (path_) free(path_);
    }
  
    mxArray *get();
    void set(const mxArray *pm);
    mxArray *curve();
    mxArray *path();
    mxArray *q();
    void run();
  
  protected:
    int discretize(State s);
    int act(int si);
    int explore(int ai);
    State step(State s, int ai, float *r);
};
