/*
 * Driver.h
 *
 *  Created on: June 15, 2017
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3LDG.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "BeamGraph.h"
#include "GreedyGraph.h"

class Driver {
  public:
    Driver() {
        _batch = 0;
        _clip = 10.0;
    }

    ~Driver() {
        _batch = 0;
        _clip = 10.0;
        //_beam_builders.clear();
    }

  public:
    Graph _cg;  // build neural graphs
    vector<Graph> _decode_cgs;
    //vector<BeamGraphBuilder> _beam_builders;
    std::vector<GreedyGraphBuilder> _greedy_builders;
    ModelParams _modelparams;  // model parameters
    HyperParams _hyperparams;

    Metric _eval;
    ModelUpdate _ada;  // model update
    //ModelUpdate _beam_ada;  // model update

    int _batch;
    bool _useBeam;
    dtype _clip;

  public:

    inline void initial() {
        if (!_hyperparams.bValid()) {
            std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
            return;
        }
        if (!_modelparams.initial(_hyperparams)) {
            std::cout << "model parameter initialization Error, Please check!" << std::endl;
            return;
        }
        _hyperparams.print();

        //_beam_builders.resize(_hyperparams.batch);
        _greedy_builders.resize(_hyperparams.batch);
        _decode_cgs.resize(_hyperparams.batch);

        dtype dropout_value = _hyperparams.dropProb;
        _hyperparams.dropProb = -1.0;
//        for (int idx = 0; idx < _hyperparams.batch; idx++) {
//            _beam_builders[idx].initial(_modelparams, _hyperparams);
//        }

        _hyperparams.dropProb = dropout_value;
        for (auto & builder : _greedy_builders) {
            builder.initial(_modelparams, _hyperparams);
        }

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
        _batch = 0;
        _useBeam = false;
    }


  public:
    dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
        _eval.reset();
        dtype cost = 0.0;
        int num = sentences.size();
        if (false) {
//            if (num > _beam_builders.size()) {
//                std::cout << "input example number is larger than predefined batch number" << std::endl;
//                return -1;
//            }

//            _cg.clearValue(true);
//            for (int idx = 0; idx < num; idx++) {
//                _beam_builders[idx].encode(&_cg, &sentences[idx]);
//            }
//            _cg.compute();

//            #pragma omp parallel for schedule(static,1)
//            for (int idx = 0; idx < num; idx++) {
//                _decode_cgs[idx].clearValue(true);
//                _beam_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx], &goldACs[idx]);
//                _eval.overall_label_count += goldACs[idx].size();
//                cost += loss_google(_beam_builders[idx], num, cpu_losses,
//                        gpu_losses);
//                _decode_cgs[idx].backward();
//            }

//            _cg.backward();
        } else {
            _cg.clearValue(true);
            for (int idx = 0; idx < num; idx++) {
                _greedy_builders.at(idx).encode(&_cg, &sentences[idx]);
            }
            _cg.compute();
            #pragma omp parallel for schedule(static,1)
            for (int idx = 0; idx < num; idx++) {
                _decode_cgs[idx].clearValue(true);
                _greedy_builders.at(idx).decode(&(_decode_cgs[idx]), &sentences[idx], &goldACs[idx]);
                _eval.overall_label_count += goldACs[idx].size();

                cost += loss_google(_greedy_builders.at(idx), num);
            }
            for (int i = 0; i < num; ++i) {
                _decode_cgs[i].backward();
            }
            _cg.backward();
        }
        return cost;
    }

    void decode(const std::vector<std::vector<string> >& sentences, vector<vector<string> >& seg_results, vector<vector<string> >& tag_results) {
        int num = sentences.size();
        if (false) {
//            if (num > _beam_builders.size()) {
//                std::cout << "input example number is larger than predefined batch number" << std::endl;
//                return;
//            }
//            _cg.clearValue();
//            for (int idx = 0; idx < num; idx++) {
//                _beam_builders[idx].encode(&_cg, &sentences[idx]);
//            }
//            _cg.compute();

//            seg_results.resize(num);
//            tag_results.resize(num);
//            #pragma omp parallel for schedule(static,1)
//            for (int idx = 0; idx < num; idx++) {
//                _decode_cgs[idx].clearValue();
//                _beam_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx]);
//                int step = _beam_builders[idx].outputs.size();
//                _beam_builders[idx].states[step - 1][0].getResults(seg_results[idx], tag_results[idx], &_hyperparams);
//            }
        } else {
            _cg.clearValue();
            for (int idx = 0; idx < num; idx++) {
                _greedy_builders.at(idx).encode(&_cg, &sentences[idx]);
            }
            _cg.compute();

            seg_results.resize(num);
            tag_results.resize(num);
            #pragma omp parallel for schedule(static,1)
            for (int idx = 0; idx < num; idx++) {
                _decode_cgs[idx].clearValue();
                _greedy_builders.at(idx).decode(&(_decode_cgs[idx]), &sentences[idx]);
                int step = _greedy_builders.at(idx).outputs.size();
                _greedy_builders.at(idx).states[step - 1].getResults(seg_results[idx], tag_results[idx], &_hyperparams);
            }
        }
    }

    void updateModel() {
        //if (_batch <= 0) return;
        if (true) {
            if (_ada._params.empty()) {
                _modelparams.exportModelParams(_ada);
            }
            //_ada.rescaleGrad(1.0 / _batch);
            //_ada.update(10);
            _ada.updateAdam(_clip);
            _batch = 0;
        } else {
//            if (_beam_ada._params.empty()) {
//                _modelparams.exportModelBeamParams(_beam_ada);
//            }
//            _beam_ada.rescaleGrad(1.0 / _batch);
//            _beam_ada.update(10);
//            _beam_ada.updateAdam(_clip);
//            _batch = 0;
        }
    }

    void writeModel();

    void loadModel();

  private:
    dtype loss_google(GreedyGraphBuilder& builder, int num) {
        std::vector<dtype*> cpu_vals, gpu_vals;
#if USE_GPU
        for (auto &vec : builder.outputs) {
            for (COutput &output : vec) {
                cpu_vals.push_back(output.in->val.v);
                gpu_vals.push_back(output.in->val.value);
            }
        }
        n3ldg_cuda::CopyFromDeviceToHost(gpu_vals, cpu_vals, gpu_vals.size(),
                1);
#endif

        int maxstep = builder.outputs.size();
        if (maxstep == 0) return 1.0;
        //_eval.correct_label_count += maxstep;
        PNode pBestNode = NULL;
        PNode pGoldNode = NULL;
        PNode pCurNode;
        dtype sum, max;
        int curcount, goldIndex;
        vector<dtype> scores;
        dtype cost = 0.0;

        for (int step = 0; step < maxstep; step++) {
            curcount = builder.outputs[step].size();
            if (curcount == 1) {
                _eval.correct_label_count++;
                continue;
            }
            max = 0.0;
            goldIndex = -1;
            pBestNode = pGoldNode = NULL;
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
                    pBestNode = pCurNode;
                }
                if (builder.outputs[step][idx].bGold) {
                    pGoldNode = pCurNode;
                    goldIndex = idx;
                }
            }

            if (goldIndex == -1) {
                std::cout << "impossible" << std::endl;
            }
            pGoldNode->loss[0] = -1.0 / num;

            max = pBestNode->val[0];
            sum = 0.0;
            scores.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] - max);
                sum += scores[idx];
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                pCurNode->loss[0] += scores[idx] / (sum * num);
            }

            if (pBestNode == pGoldNode)_eval.correct_label_count++;
            //_eval.overall_label_count++;

            cost += -log(scores[goldIndex] / sum);

            if (std::isnan(cost)) {
                std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
            }

            _batch++;
        }

#if USE_GPU
        std::vector<dtype*> cpu_losses, gpu_losses;
        for (auto &vec : builder.outputs) {
            for (COutput &output : vec) {
                cpu_losses.push_back(output.in->loss.v);
                gpu_losses.push_back(output.in->loss.value);
            }
        }

        n3ldg_cuda::CopyFromHostToDevice(cpu_losses, gpu_losses,
                gpu_losses.size(), 1);
#endif
        return cost;
    }

    dtype loss_google(BeamGraphBuilder& builder, int num) {
        int maxstep = builder.outputs.size();
        if (maxstep == 0) return 1.0;
        //_eval.correct_label_count += maxstep;
        PNode pBestNode = NULL;
        PNode pGoldNode = NULL;
        PNode pCurNode;
        dtype sum, max;
        int curcount, goldIndex;
        vector<dtype> scores;
        dtype cost = 0.0;

        for (int step = maxstep-1; step < maxstep; step++) {
            curcount = builder.outputs[step].size();
            if (curcount == 1) {
                _eval.correct_label_count++;
                continue;
            }
            max = 0.0;
            goldIndex = -1;
            pBestNode = pGoldNode = NULL;
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
                    pBestNode = pCurNode;
                }
                if (builder.outputs[step][idx].bGold) {
                    pGoldNode = pCurNode;
                    goldIndex = idx;
                }
            }

            if (goldIndex == -1) {
                std::cout << "impossible" << std::endl;
            }
            pGoldNode->loss[0] = -1.0 / num;

            max = pBestNode->val[0];
            sum = 0.0;
            scores.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] - max);
                sum += scores[idx];
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                pCurNode->loss[0] += scores[idx] / (sum * num);
            }

            if (pBestNode == pGoldNode)_eval.correct_label_count++;
            //_eval.overall_label_count++;

            cost += -log(scores[goldIndex] / sum);

            if (std::isnan(cost)) {
                std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
            }

            _batch++;
        }

        return cost;
    }

  public:
    inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
        _ada._alpha = adaAlpha;
        _ada._eps = adaEps;
        _ada._reg = nnRegular;

//        _beam_ada._alpha = adaAlpha * 0.1;
//        _beam_ada._eps = adaEps;
//        _beam_ada._reg = nnRegular;
    }

    //useBeam = true, beam searcher
    inline void setGraph(bool useBeam) {
        _useBeam = useBeam;
    }

    inline void setClip(dtype clip) {
        _clip = clip;
    }

};

#endif /* SRC_Driver_H_ */
