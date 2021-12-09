/*
 * connector.h
 *
 *  Created on: Aug 3, 2012
 *      Author: wcaarls
 */

#ifndef MPRL_ENV_MATLAB_MEMSTRING_H_
#define MPRL_ENV_MATLAB_MEMSTRING_H_

#include <string>
#include <mex.h>

class MexMemString
{
  private:
    char *string_;

  public:
    MexMemString() : string_(NULL) { }
    ~MexMemString()
    {
      if (string_)
        mxFree(string_);
    }

    operator char*()
    {
      return string_;
    }

    operator std::string()
    {
      return std::string(string_);
    }

    MexMemString &operator=(char *rhs)
    {
      if (string_)
        mxFree(string_);

      string_ = rhs;

      return *this;
    }
};

#endif /* MPRL_ENV_MATLAB_MEMSTRING_H_ */
