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

#include "tsid/robots/robot-wrapper.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

using namespace se3;
using namespace tsid::math;

namespace tsid
{
  namespace robots
  {
    
    RobotWrapper::RobotWrapper(const std::string & filename,
                               const std::vector<std::string> & ,
                               bool verbose)
    : m_verbose(verbose)
    {
      se3::urdf::buildModel(filename, m_model, m_verbose);
      m_model_filename = filename;
      m_rotor_inertias.setZero(m_model.nv);
      m_gear_ratios.setZero(m_model.nv);
      m_Md.setZero(m_model.nv);
      m_M.setZero(m_model.nv, m_model.nv);
    }
    
    RobotWrapper::RobotWrapper(const std::string & filename,
                               const std::vector<std::string> & ,
                               const se3::JointModelVariant & rootJoint,
                               bool verbose)
    : m_verbose(verbose)
    {
      se3::urdf::buildModel(filename, rootJoint, m_model, m_verbose);
      m_model_filename = filename;
      m_rotor_inertias.setZero(m_model.nv-6);
      m_gear_ratios.setZero(m_model.nv-6);
      m_Md.setZero(m_model.nv-6);
      m_M.setZero(m_model.nv, m_model.nv);
    }
    
    int RobotWrapper::nq() const { return m_model.nq; }
    int RobotWrapper::nv() const { return m_model.nv; }
    
    const Model & RobotWrapper::model() const { return m_model; }
    Model & RobotWrapper::model() { return m_model; }
    
    void RobotWrapper::computeAllTerms(Data & data, const Vector & q, const Vector & v) const
    {
      se3::computeAllTerms(m_model, data, q, v);
      data.M.triangularView<Eigen::StrictlyLower>()
            = data.M.transpose().triangularView<Eigen::StrictlyLower>();
      // computeAllTerms does not compute the com acceleration, so we need to call centerOfMass
      se3::centerOfMass<true,true,true>(m_model, data, false);
      se3::framesForwardKinematics(m_model, data);
      se3::centerOfMass(m_model, data, q, v, Eigen::VectorXd::Zero(nv()));
      se3::ccrba(m_model, data, q, v);
    }
    
    const Vector & RobotWrapper::rotor_inertias() const
    {
      return m_rotor_inertias;
    }
    const Vector & RobotWrapper::gear_ratios() const
    {
      return m_gear_ratios;
    }
    
    bool RobotWrapper::rotor_inertias(ConstRefVector rotor_inertias)
    {
      assert(rotor_inertias.size()==m_rotor_inertias.size());
      m_rotor_inertias = rotor_inertias;
      updateMd();
      return true;
    }
    
    bool RobotWrapper::gear_ratios(ConstRefVector gear_ratios)
    {
      assert(gear_ratios.size()==m_gear_ratios.size());
      m_gear_ratios = gear_ratios;
      updateMd();
      return true;
    }
    
    void RobotWrapper::updateMd()
    {
      m_Md = m_gear_ratios.cwiseProduct(m_gear_ratios.cwiseProduct(m_rotor_inertias));
    }
    
    void RobotWrapper::com(const Data & data,
                           RefVector com_pos,
                           RefVector com_vel,
                           RefVector com_acc) const
    {
      com_pos = data.com[0];
      com_vel = data.vcom[0];
      com_acc = data.acom[0];
    }
    
    const Vector3 & RobotWrapper::com(const Data & data) const
    {
      return data.com[0];
    }
    
    const Vector3 & RobotWrapper::com_vel(const Data & data) const
    {
      return data.vcom[0];
    }
    
    const Vector3 & RobotWrapper::com_acc(const Data & data) const
    {
      return data.acom[0];
    }
    
    const Matrix3x & RobotWrapper::Jcom(const Data & data) const
    {
      return data.Jcom;
    }
    
    const Matrix & RobotWrapper::mass(const Data & data)
    {
      m_M = data.M;
      m_M.diagonal().tail(m_model.nv-6) += m_Md;
      return m_M;
    }
    
    const Vector & RobotWrapper::nonLinearEffects(const Data & data) const
    {
      return data.nle;
    }
    
    const SE3 & RobotWrapper::position(const Data & data,
                                       const Model::JointIndex index) const
    {
      return data.oMi[index];
    }
    
    const Motion & RobotWrapper::velocity(const Data & data,
                                          const Model::JointIndex index) const
    {
      return data.v[index];
    }
    
    const Motion & RobotWrapper::acceleration(const Data & data,
                                              const Model::JointIndex index) const
    {
      return data.a[index];
    }
    
    void RobotWrapper::jacobianWorld(const Data & data,
                                     const Model::JointIndex index,
                                     Data::Matrix6x & J) const
    {
      return se3::getJacobian<se3::WORLD>(m_model, data, index, J);
    }
    
    void RobotWrapper::jacobianLocal(const Data & data,
                                     const Model::JointIndex index,
                                     Data::Matrix6x & J) const
    {
      return se3::getJacobian<se3::LOCAL>(m_model, data, index, J);
    }
    
    SE3 RobotWrapper::framePosition(const Data & data,
                                    const Model::FrameIndex index) const
    {
      const Frame & f = m_model.frames[index];
      return data.oMi[f.parent].act(f.placement);
    }
    
    void RobotWrapper::framePosition(const Data & data,
                                     const Model::FrameIndex index,
                                     SE3 & framePosition) const
    {
      const Frame & f = m_model.frames[index];
      framePosition = data.oMi[f.parent].act(f.placement);
    }
    
    Motion RobotWrapper::frameVelocity(const Data & data,
                                       const Model::FrameIndex index) const
    {
      const Frame & f = m_model.frames[index];
      return f.placement.actInv(data.v[f.parent]);
    }
    
    void RobotWrapper::frameVelocity(const Data & data,
                                     const Model::FrameIndex index,
                                     Motion & frameVelocity) const
    {
      const Frame & f = m_model.frames[index];
      frameVelocity = f.placement.actInv(data.v[f.parent]);
    }
    
    Motion RobotWrapper::frameAcceleration(const Data & data,
                                           const Model::FrameIndex index) const
    {
      const Frame & f = m_model.frames[index];
      return f.placement.actInv(data.a[f.parent]);
    }
    
    void RobotWrapper::frameAcceleration(const Data & data,
                                         const Model::FrameIndex index,
                                         Motion & frameAcceleration) const
    {
      const Frame & f = m_model.frames[index];
      frameAcceleration = f.placement.actInv(data.a[f.parent]);
    }
    
    Motion RobotWrapper::frameClassicAcceleration(const Data & data,
                                                  const Model::FrameIndex index) const
    {
      const Frame & f = m_model.frames[index];
      Motion a = f.placement.actInv(data.a[f.parent]);
      Motion v = f.placement.actInv(data.v[f.parent]);
      a.linear() += v.angular().cross(v.linear());
      return a;
    }
    
    void RobotWrapper::frameClassicAcceleration(const Data & data,
                                                const Model::FrameIndex index,
                                                Motion & frameAcceleration) const
    {
      const Frame & f = m_model.frames[index];
      frameAcceleration = f.placement.actInv(data.a[f.parent]);
      Motion v = f.placement.actInv(data.v[f.parent]);
      frameAcceleration.linear() += v.angular().cross(v.linear());
    }
    
    void RobotWrapper::frameJacobianWorld(const Data & data,
                                          const Model::FrameIndex index,
                                          Data::Matrix6x & J) const
    {
      return se3::getJacobian<se3::WORLD>(m_model, data, m_model.frames[index].parent, J);
    }
    
    void RobotWrapper::frameJacobianLocal(const Data & data,
                                          const Model::FrameIndex index,
                                          Data::Matrix6x & J) const
    {
      return se3::getFrameJacobian(m_model, data, index, J);
    }
    const Matrix6x & RobotWrapper::momentumJacobian(const Data & data) const
    {
      return data.Ag;
    }
    
    //    const Vector3 & com(Data & data,const Vector & q,
    //                        const bool computeSubtreeComs = true,
    //                        const bool updateKinematics = true)
    //    {
    //      return se3::centerOfMass(m_model, data, q, computeSubtreeComs, updateKinematics);
    //    }
    //    const Vector3 & com(Data & data, const Vector & q, const Vector & v,
    //                 const bool computeSubtreeComs = true,
    //                 const bool updateKinematics = true)
    //    {
    //      return se3::centerOfMass(m_model, data, q, v, computeSubtreeComs, updateKinematics);
    //    }
    
  } // namespace robots
}
