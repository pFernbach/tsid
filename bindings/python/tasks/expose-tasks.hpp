//
// Copyright (c) 2018 CNRS
//
// This file is part of tsid
// tsid is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
// tsid is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// tsid If not, see
// <http://www.gnu.org/licenses/>.
//

#ifndef __tsid_python_expose_tasks_hpp__
#define __tsid_python_expose_tasks_hpp__

#include "tsid/bindings/python/tasks/task-com-equality.hpp"
#include "tsid/bindings/python/tasks/task-se3-equality.hpp"
#include "tsid/bindings/python/tasks/task-joint-posture.hpp"
#include "tsid/bindings/python/tasks/task-posture.hpp"
#include "tsid/bindings/python/tasks/task-am-equality.hpp"

namespace tsid
{
  namespace python
  {
    void exposeTaskComEquality();
    void exposeTaskSE3Equality();
    void exposeTaskJointPosture();
    void exposeTaskPosture();
    void exposeTaskAMEquality();

    inline void exposeTasks()
    {
      exposeTaskComEquality();
      exposeTaskSE3Equality();
      exposeTaskJointPosture();
      exposeTaskPosture();
      exposeTaskAMEquality();
    }
    
  } // namespace python
} // namespace tsid
#endif // ifndef __tsid_python_expose_tasks_hpp__
