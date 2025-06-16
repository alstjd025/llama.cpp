#include "llama-batch.h"

#include "llama-impl.h"
#include "llama-vocab.h"
#include "llama-memory.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <sstream>

llama_batch_allocr::llama_batch_allocr() {
    const char * LLAMA_BATCH_DEBUG = getenv("LLAMA_BATCH_DEBUG");
    debug = LLAMA_BATCH_DEBUG ? atoi(LLAMA_BATCH_DEBUG) : 0;

    seq_pos.resize(LLAMA_MAX_SEQ);
    seq_cpl.resize(LLAMA_MAX_SEQ);
    for (auto & cur : seq_cpl) {
        cur.resize(LLAMA_MAX_SEQ);
    }
}

bool llama_batch_allocr::init(
        const llama_batch & batch_inp,
        const llama_vocab & vocab,
        const llama_memory_i * memory,
        uint32_t n_embd,
        bool output_all) {
    clear();

    batch = batch_inp;

    GGML_ASSERT(batch.n_tokens > 0);

    //
    // validate input batch
    //

    if (batch.token) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return false;
            }
        }
    }

    if (batch.seq_id) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                if (batch.seq_id && (batch.seq_id[i][s] < 0 || batch.seq_id[i][s] >= LLAMA_MAX_SEQ)) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id[%d][%d] = %d > %d\n", __func__, i, s, batch.seq_id[i][s], LLAMA_MAX_SEQ);
                    return false;
                }
            }
        }
    }

    //
    // auto-generate missing fields
    //

    if (!batch.n_seq_id) {
        n_seq_id.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            n_seq_id[i] = seq_id_0.size();
        }
        batch.n_seq_id = n_seq_id.data();
    }

    if (!batch.seq_id) {
        seq_id.resize(batch.n_tokens + 1);
        seq_id[batch.n_tokens] = NULL;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            seq_id[i] = seq_id_0.data();
        }
        batch.seq_id = seq_id.data();
    }

    if (!batch.pos) {
        pos.resize(batch.n_tokens);

        // initialize the starting position for each sequence based on the positions in the memory
        llama_pos p0[LLAMA_MAX_SEQ];
        for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (!memory) {
                p0[s] = 0;
            } else {
                p0[s] = memory->seq_pos_max(s) + 1;
            }
        }

        for (int32_t i = 0; i < batch.n_tokens; i++) {
            const llama_seq_id seq_id = batch.seq_id[i][0];

            pos[i] = p0[seq_id];

            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                p0[batch.seq_id[i][s]] = pos[i] + 1;
            }
        }

        batch.pos = pos.data();
    }

    if (!batch.logits) {
        if (output_all) {
            // return the output for all tokens
            output.resize(batch.n_tokens, true);
        } else {
            // return the output only for the last token
            output.resize(batch.n_tokens, false);
            output[output.size() - 1] = true;
        }

        batch.logits = output.data();
    } else if (output_all) {
        bool warn = false;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] == 0) {
                warn = true;
            }
        }

        if (warn) {
            LLAMA_LOG_WARN("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n", __func__);

            output.resize(batch.n_tokens, true);
            batch.logits = output.data();
        }
    }

    //
    // compute stats
    //

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        n_outputs += batch.logits[i] != 0;
    }

    this->n_embd = n_embd;

    // determine coupled sequences
    // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            seq_pos[batch.seq_id[i][s]].insert(batch.pos[i]);

            if (s > 0) {
                const llama_seq_id s0 = batch.seq_id[i][0];
                const llama_seq_id s1 = batch.seq_id[i][s];

                // mark that sequence s1 is coupled to s0
                seq_cpl[s1][s0] = true;

                // note: the other way around is not necessary for now
                //seq_cpl[s0][s1] = true;
            }
        }
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        seq_set_t cur;
        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            cur.set(batch.seq_id[i][s]);
        }

        seq_set.push_back(cur);
        seq_set_map[cur].push_back(i);
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: input batch info:\n", __func__);
        LLAMA_LOG_DEBUG("%s:   n_tokens  = %d\n", __func__,          batch.n_tokens);
        LLAMA_LOG_DEBUG("%s:   token     = %p\n", __func__, (void *) batch.token);
        LLAMA_LOG_DEBUG("%s:   embd      = %p\n", __func__, (void *) batch.embd);
        LLAMA_LOG_DEBUG("%s:   pos       = %p\n", __func__, (void *) batch.pos);
        LLAMA_LOG_DEBUG("%s:   n_seq_id  = %p\n", __func__, (void *) batch.n_seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id    = %p\n", __func__, (void *) batch.seq_id);
        LLAMA_LOG_DEBUG("%s:   logits    = %p\n", __func__, (void *) batch.logits);
        LLAMA_LOG_DEBUG("%s:   n_outputs = %d\n", __func__, n_outputs);

        if (debug > 1) {
            int seq_id_max = 0;
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                    for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                        seq_id_max = std::max(seq_id_max, batch.seq_id[i][s]);
                    }
                }
            }
            ++seq_id_max;

            LLAMA_LOG_DEBUG("%s:   token     = [\n", __func__);
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                std::vector<int8_t> seq_id(seq_id_max);

                for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                    seq_id[batch.seq_id[i][s]] = 1;
                }

                std::stringstream ss;
                for (int s = 0; s < seq_id_max; ++s) {
                    if (seq_id[s]) {
                        ss << s%10;
                    } else {
                        ss << ".";
                    }
                }

                LLAMA_LOG_DEBUG("%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                        __func__, i, batch.token[i], vocab.token_to_piece(batch.token[i]).c_str(),
                        batch.pos[i], batch.n_seq_id[i], ss.str().c_str(), batch.logits[i]);
            }
            LLAMA_LOG_DEBUG("%s:   ]\n", __func__);

            LLAMA_LOG_DEBUG("%s:   seq       = [\n", __func__);
            for (int s0 = 0; s0 < (int) seq_pos.size(); ++s0) {
                if (seq_pos[s0].empty()) {
                    continue;
                }

                std::stringstream ss;
                for (int s1 = 0; s1 < (int) seq_cpl[s0].size(); ++s1) {
                    if (seq_cpl[s0][s1]) {
                        ss << s1 << " ";
                    }
                }

                LLAMA_LOG_DEBUG("%s:  %4d: pos = [%4d, %4d], cpl = %s\n",
                        __func__, s0, seq_pos_min(s0), seq_pos_max(s0), ss.str().empty() ? "-" : ss.str().c_str());
            }
            LLAMA_LOG_DEBUG("%s:   ]\n", __func__);
        }
    }

    //
    // consistency checks
    //

    for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos[s].empty()) {
            continue;
        }

        if (memory && seq_pos_min(s) != memory->seq_pos_max(s) + 1) {
            LLAMA_LOG_ERROR("%s: sequence %d does not start from the last position stored in the memory\n", __func__, s);
            return false;
        }

        if (seq_pos_max(s) - seq_pos_min(s) + 1 > (int) seq_pos[s].size()) {
            LLAMA_LOG_ERROR("%s: sequence %d positions are not continuous\n", __func__, s);
            return false;
        }
    }

    if (memory) {
        for (int32_t s0 = 0; s0 < LLAMA_MAX_SEQ; ++s0) {
            for (int32_t s1 = 0; s1 < LLAMA_MAX_SEQ; ++s1) {
                if (seq_cpl[s0][s1]) {
                    if (memory->seq_pos_min(s0) != memory->seq_pos_min(s1) ||
                        memory->seq_pos_max(s0) != memory->seq_pos_max(s1)) {
                        LLAMA_LOG_ERROR("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", __func__, s0, s1);
                        return false;
                    }
                }
            }
        }
    }

    {
        seq_set_t cur_seq_set[LLAMA_MAX_SEQ];
        for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            cur_seq_set[s].set();
        }

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cur_seq_set[seq_id] &= seq_set[i];

                if (cur_seq_set[seq_id].none()) {
                    LLAMA_LOG_ERROR("%s: sequence %d belongs to incompatible sequence sets\n", __func__, seq_id);
                    return false;
                }
            }
        }
    }

    // TODO: check that positions are increasing

    split_reset();

    return true;
}

llama_ubatch llama_batch_allocr::reserve_one(uint32_t n_tokens) {
    clear();
    split_reset();

    ubatches.emplace_back();

    auto & ubatch = ubatches.back();

    ubatch.token   .resize(n_tokens);
    ubatch.embd    .clear();
    ubatch.pos     .resize(n_tokens);
    ubatch.n_seq_id.resize(n_tokens);
    ubatch.seq_id  .resize(n_tokens);
    ubatch.output  .resize(n_tokens);

    llama_ubatch res {
        /*.equal_seqs   =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_tokens,
        /*.n_seqs       =*/ 1,

        /*.token        =*/ ubatch.token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ ubatch.pos.data(),
        /*.n_seq_id     =*/ ubatch.n_seq_id.data(),
        /*.seq_id       =*/ ubatch.seq_id.data(),
        /*.output       =*/ ubatch.output.data(),
    };

    return res;
}

const llama_batch & llama_batch_allocr::get_batch() const {
    return batch;
}

uint32_t llama_batch_allocr::get_n_tokens() const {
    return batch.n_tokens;
}

uint32_t llama_batch_allocr::get_n_outputs() const {
    return n_outputs;
}

std::vector<int32_t> & llama_batch_allocr::get_out_ids() {
    return out_ids;
}

llama_pos llama_batch_allocr::seq_pos_min(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].begin();
}

llama_pos llama_batch_allocr::seq_pos_max(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].rbegin();
}

void llama_batch_allocr::split_reset() {
    out_ids.clear();

    used.clear();
    used.resize(get_n_tokens(), false);

    ubatches.clear();
}

llama_ubatch llama_batch_allocr::split_simple(uint32_t n_ubatch) {
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    if (cur_idx >= used.size()) {
        return {};
    }

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;

        ++cur_idx;

        if (cur_idx >= used.size()) {
            break;
        }

        if (idxs.size() >= n_ubatch) {
            break;
        }
    }

    return add_ubatch(idxs, idxs.size(), false);
}

llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch) {
    std::vector<seq_set_t> cur_seq_set;

    // determine the sequence sets participating in this ubatch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (used[i]) {
            continue;
        }

        bool add = true;

        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            // no overlap with existing sequence sets:
            if (!(cur_seq_set[s] & seq_set[i]).none()) {
                add = false;
                break;
            }
        }

        if (add) {
            cur_seq_set.push_back(seq_set[i]);

            if (cur_seq_set.size() > n_ubatch) {
                break;
            }
        }
    }

    const uint32_t n_seqs = cur_seq_set.size();

    if (n_seqs == 0) {
        return {};
    }

    std::vector<int32_t> cur_idx(n_seqs, 0);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        while (used[seq_set_map[cur_seq_set[s]][cur_idx[s]]]) {
            ++cur_idx[s];
        }
    }

    std::vector<idx_vec_t> idxs_per_seq(n_seqs);

    while (true) {
        bool can_expand = true;

        for (uint32_t s = 0; s < n_seqs; ++s) {
            if (cur_idx[s] >= (int32_t) seq_set_map[cur_seq_set[s]].size()) {
                can_expand = false;
                break;
            }
        }

        if (!can_expand) {
            break;
        }

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const int32_t idx = seq_set_map[cur_seq_set[s]][cur_idx[s]];
            idxs_per_seq[s].push_back(idx);

            used[idx] = true;

            ++cur_idx[s];
        }

        if  ((idxs_per_seq[0].size() + 1)*n_seqs > n_ubatch) {
            break;
        }
    }

    std::vector<int32_t> idxs;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        idxs.insert(idxs.end(), idxs_per_seq[s].begin(), idxs_per_seq[s].end());
    }

    return add_ubatch(idxs, n_seqs, true);
}

llama_ubatch llama_batch_allocr::split_seq(uint32_t n_ubatch) {
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    if (cur_idx >= used.size()) {
        return {};
    }

    auto cur_seq_set = seq_set[cur_idx];

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;

        if (idxs.size() >= n_ubatch) {
            break;
        }

        do {
            ++cur_idx;
        } while (cur_idx < get_n_tokens() && (used[cur_idx] || ((cur_seq_set & seq_set[cur_idx]) != seq_set[cur_idx])));

        if (cur_idx == get_n_tokens()) {
            break;
        }

        cur_seq_set = seq_set[cur_idx];
    }

    return add_ubatch(idxs, 1, true);
}

void llama_batch_allocr::clear() {
    n_outputs = 0;

    batch = {};

    pos     .clear();
    n_seq_id.clear();
    seq_id  .clear();
    output  .clear();

    for (auto & cur : seq_pos) {
        cur.clear();
    }

    for (auto & cur : seq_cpl) {
        std::fill(cur.begin(), cur.end(), false);
    }

    seq_set.clear();

    seq_set_map.clear();
}

llama_ubatch llama_batch_allocr::add_ubatch(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    const uint32_t n_tokens = idxs.size();

    LLAMA_LOG_DEBUG("add_ubatch: n_tokens = %d, n_seqs = %d, equal_seqs = %d", n_tokens, n_seqs, equal_seqs);

    assert(n_tokens%n_seqs == 0);

    ubatches.emplace_back();

    auto & ubatch = ubatches.back();

    ubatch.token   .resize(n_tokens);
    ubatch.embd    .resize((int64_t) n_tokens*n_embd);
    ubatch.pos     .resize(n_tokens);
    ubatch.n_seq_id.resize(n_tokens);
    ubatch.seq_id  .resize(n_tokens);
    ubatch.output  .resize(n_tokens);

    for (size_t i = 0; i < idxs.size(); ++i) {
        if (batch.token) {
            ubatch.token[i] = batch.token[idxs[i]];
        }

        if (batch.embd) {
            memcpy(ubatch.embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        ubatch.pos[i]      = batch.pos[idxs[i]];
        ubatch.n_seq_id[i] = batch.n_seq_id[idxs[i]];
        ubatch.seq_id[i]   = batch.seq_id[idxs[i]];
        ubatch.output[i]   = batch.logits[idxs[i]];

        if (ubatch.output[i]) {
            out_ids.push_back(idxs[i]);
        }
    }

    llama_ubatch res {
        /*.equal_seqs   =*/ equal_seqs,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_tokens/n_seqs,
        /*.n_seqs       =*/ n_seqs,

        /*.token        =*/ batch.token ? ubatch.token.data() : nullptr,
        /*.embd         =*/ batch.embd ? ubatch.embd.data() : nullptr,
        /*.pos          =*/ ubatch.pos.data(),
        /*.n_seq_id     =*/ ubatch.n_seq_id.data(),
        /*.seq_id       =*/ ubatch.seq_id.data(),
        /*.output       =*/ ubatch.output.data(),
    };

    LLAMA_LOG_DEBUG("%s: added ubatch of size %d\n", __func__, res.n_tokens);

    return res;
}

//
// interface implementation
//

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens =*/ n_tokens,
        /*tokens   =*/ tokens,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = {
        /*n_tokens =*/ 0,
        /*tokens   =*/ nullptr,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens_alloc);
    }

    batch.pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens_alloc);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;

    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens_alloc);

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}
