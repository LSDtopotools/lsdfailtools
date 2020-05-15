#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <cmath>

namespace py = pybind11;

#include "LSDIversonWrapper.hpp"

// Python Module and Docstrings

PYBIND11_MODULE(lsdfailtools, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        TO DO

        .. currentmodule:: lsdfailtools

        .. autosummary::
           :toctree: _generate

           example1
           example2
           readme_example1
           vectorize_example1
    )pbdoc";

     py::class_<lsdiversonwrapper>(m, "lsdiverson",py::dynamic_attr())
      .def(py::init<>())
      //.def(py::init([](/*param*/){return std::unique_ptr<LSDDEM_xtensor>(new LSDDEM_xtensor(/*param without identifier*/)); })) // <- template for new constructors
      .def(py::init([](float talpha,float tD_0,float tK_sat,float tD_hat,float td,float tbeta,float tIz_over_K_steady,
        float tfriction_angle,float tcohesion,float tweight_of_water,float tweight_of_soil, float tminimum_depth){
        return std::unique_ptr<lsdiversonwrapper>(new lsdiversonwrapper( talpha, tD_0, tK_sat, tD_hat, td, tbeta, tIz_over_K_steady,
       tfriction_angle, tcohesion, tweight_of_water, tweight_of_soil, tminimum_depth)); }))
      .def("set_duration_intensity",&lsdiversonwrapper::get_duration_intensity_from_preprocessed_input)
      .def("ScanTimeseriesForFailure",&lsdiversonwrapper::ScanTimeseriesForFailure)
      .def("set_depths_vector",&lsdiversonwrapper::set_depths_vector)
      .def_readwrite("output_times", &lsdiversonwrapper::output_times)
      .def_readwrite("output_depthsFS", &lsdiversonwrapper::output_depthsFS)
      .def_readwrite("output_minFS", &lsdiversonwrapper::output_minFS)
      .def_readwrite("output_PsiFS", &lsdiversonwrapper::output_PsiFS)
      .def_readwrite("output_durationFS", &lsdiversonwrapper::output_durationFS)
      .def_readwrite("output_intensityFS", &lsdiversonwrapper::output_intensityFS)
      .def_readwrite("output_failure_times", &lsdiversonwrapper::output_failure_times)
      .def_readwrite("output_failure_mindepths", &lsdiversonwrapper::output_failure_mindepths)
      .def_readwrite("output_failure_maxdepths", &lsdiversonwrapper::output_failure_maxdepths)
      .def_readwrite("output_Psi_timedepth", &lsdiversonwrapper::output_Psi_timedepth)
      .def_readwrite("output_FS_timedepth", &lsdiversonwrapper::output_FS_timedepth)      
      ;

}
