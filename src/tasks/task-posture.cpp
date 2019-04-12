//
// Copyright (c) 2017 CNRS
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

#include <tsid/tasks/task-posture.hpp>
#include "tsid/robots/robot-wrapper.hpp"
#include "tsid/math/utils.hpp"
namespace tsid
{
  namespace tasks
  {
    using namespace math;
    using namespace trajectories;
    using namespace se3;

    TaskPosture::TaskPosture(const std::string & name,
                                     RobotWrapper & robot):
      TaskMotion(name, robot),
      m_ref(robot.nv()),
      m_constraint(name, robot.nv(), robot.nv())
    {
      m_Kp.setZero(robot.nv());
      m_Kd.setZero(robot.nv());
      m_p_error.setZero(robot.nv());
      m_v_error.setZero(robot.nv());
      Vector m = Vector::Ones(robot.nv());
      mask(m);
    }

    const Vector & TaskPosture::mask() const
    {
      return m_mask;
    }

    void TaskPosture::mask(const Vector & m)
    {
      assert(m.size()==m_robot.nv());
      m_mask = m;
      const Vector::Index dim = static_cast<Vector::Index>(m.sum());
      Matrix S = Matrix::Zero(dim, m_robot.nv());
      m_activeAxes.resize(dim);
      unsigned int j=0;
      for(unsigned int i=0; i<m.size(); i++)
        if(m(i)!=0.0)
        {
          assert(m(i)==1.0);
          S(j, i) = 1.0;
          m_activeAxes(j) = i;
          j++;
        }
      m_constraint.resize((unsigned int)dim, m_robot.nv());
      m_constraint.setMatrix(S);
    }

    int TaskPosture::dim() const
    {
      return (int)m_mask.sum();
    }

    const Vector & TaskPosture::Kp(){ return m_Kp; }

    const Vector & TaskPosture::Kd(){ return m_Kd; }

    void TaskPosture::Kp(ConstRefVector Kp)
    {
      assert(Kp.size()==m_robot.nv());
      m_Kp = Kp;
    }

    void TaskPosture::Kd(ConstRefVector Kd)
    {
      assert(Kd.size()==m_robot.nv());
      m_Kd = Kd;
    }

    void TaskPosture::setReference(const TrajectorySample & ref)
    {
      assert(ref.pos.size()==m_robot.nq());
      assert(ref.vel.size()==m_robot.nv());
      assert(ref.acc.size()==m_robot.nv());
      m_ref = ref;
    }

    const TrajectorySample & TaskPosture::getReference() const
    {
      return m_ref;
    }

    const Vector & TaskPosture::getDesiredAcceleration() const
    {
      return m_a_des;
    }

    Vector TaskPosture::getAcceleration(ConstRefVector dv) const
    {
      return m_constraint.matrix()*dv;
    }

    const Vector & TaskPosture::position_error() const
    {
      return m_p_error;
    }

    const Vector & TaskPosture::velocity_error() const
    {
      return m_v_error;
    }

    const Vector & TaskPosture::position() const
    {
      return m_p;
    }

    const Vector & TaskPosture::velocity() const
    {
      return m_v;
    }

    const Vector & TaskPosture::position_ref() const
    {
      return m_ref.pos;
    }

    const Vector & TaskPosture::velocity_ref() const
    {
      return m_ref.vel;
    }

    const ConstraintBase & TaskPosture::getConstraint() const
    {
      return m_constraint;
    }

    const ConstraintBase & TaskPosture::compute(const double ,
                                                    ConstRefVector q,
                                                    ConstRefVector v,
                                                    const Data & )
    {
      // Compute errors
      m_p = q;
      m_v = v;
      se3::SE3 M_ff, M_ff_des;
      se3::Motion error_ff;

      XYZQUATToSE3(m_p.head<7>(), M_ff);
      XYZQUATToSE3(m_ref.pos.head<7>(), M_ff_des);
      errorInSE3(M_ff, M_ff_des, error_ff);

      m_p_error.head(6) = error_ff.toVector();

      m_p_error.tail(m_robot.nv()-6) = m_p.tail(m_robot.nv()-6) - m_ref.pos.tail(m_robot.nv()-6);
      m_v_error = m_v - m_ref.vel;
      
      
      m_a_des = - m_Kp.cwiseProduct(m_p_error)
                - m_Kd.cwiseProduct(m_v_error)
                + m_ref.acc;

      for(unsigned int i=0; i<m_activeAxes.size(); i++)
        m_constraint.vector()(i) = m_a_des(m_activeAxes(i));
      return m_constraint;
    }
  }
}
