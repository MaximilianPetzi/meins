#pragma once
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "sparse_matrix.hpp"

#include "pop0.hpp"
#include "pop0.hpp"



extern PopStruct0 pop0;
extern PopStruct0 pop0;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj5: pop0 -> pop0 with target exc
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct5 : LILMatrix<int> {
    ProjStruct5() : LILMatrix<int>( 200, 200) {
    }


    void fixed_probability_pattern(std::vector<int> post_ranks, std::vector<int> pre_ranks, double p, double w_dist_arg1, double w_dist_arg2, double d_dist_arg1, double d_dist_arg2, bool allow_self_connections) {
        static_cast<LILMatrix<int>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng[0]);

        w = init_matrix_variable_normal<double>(w_dist_arg1, w_dist_arg2, rng[0]);

    }





    // Number of dendrites
    int size;

    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;





    // Global parameter eta
    double  eta ;

    // Global parameter learning_phase
    double  learning_phase ;

    // Global parameter error
    double  error ;

    // Global parameter mean_error
    double  mean_error ;

    // Global parameter max_weight_change
    double  max_weight_change ;

    // Local parameter trace
    std::vector< std::vector<double > > trace;

    // Local parameter delta_w
    std::vector< std::vector<double > > delta_w;

    // Local parameter w
    std::vector< std::vector<double > > w;




    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct5::init_projection()" << std::endl;
    #endif

        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;



        // Global parameter eta
        eta = 0.0;

        // Global parameter learning_phase
        learning_phase = 0.0;

        // Global parameter error
        error = 0.0;

        // Global parameter mean_error
        mean_error = 0.0;

        // Global parameter max_weight_change
        max_weight_change = 0.0;

        // Local variable trace
        trace = init_matrix_variable<double>(static_cast<double>(0.0));

        // Local variable delta_w
        delta_w = init_matrix_variable<double>(static_cast<double>(0.0));




    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {

    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct5::compute_psp()" << std::endl;
    #endif
        int nb_post; int rk_post; int rk_pre; double sum;

        if (_transmission && pop0._active){


            nb_post = post_rank.size();

            for(int i = 0; i < nb_post; i++) {
                sum = 0.0;
                for(int j = 0; j < pre_rank[i].size(); j++) {
                    sum += pop0.r[pre_rank[i][j]]*w[i][j] ;
                }
                pop0._sum_exc[post_rank[i]] += sum;
            }

        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct5::update_synapse()" << std::endl;
    #endif

        int rk_post, rk_pre;
        double _dt = dt * _update_period;

        // Check periodicity
        if(_transmission && _update && pop0._active && ( (t - _update_offset)%_update_period == 0L) ){
            // Global variables


            // Local variables
            for(int i = 0; i < post_rank.size(); i++){
                rk_post = post_rank[i]; // Get postsynaptic rank
                // Semi-global variables

                // Local variables
                for(int j = 0; j < pre_rank[i].size(); j++){
                    rk_pre = pre_rank[i][j]; // Get presynaptic rank

                    // trace += if learning_phase < 0.5: power(pre.rprev * (post.delta_x), 3) else: 0.0
                    trace[i][j] += (learning_phase < 0.5 ? power(pop0.delta_x[post_rank[i]]*pop0.rprev[pre_rank[i][j]], 3) : 0.0);


                    // delta_w = if learning_phase > 0.5: eta * trace * (mean_error) * (error - mean_error) else: 0.0
                    delta_w[i][j] = (learning_phase > 0.5 ? eta*mean_error*trace[i][j]*(error - mean_error) : 0.0);
                    if(delta_w[i][j] < -max_weight_change)
                        delta_w[i][j] = -max_weight_change;
                    if(delta_w[i][j] > max_weight_change)
                        delta_w[i][j] = max_weight_change;


                    // w -= if learning_phase > 0.5: delta_w else: 0.0
                    if(_plasticity){
                    w[i][j] -= (learning_phase > 0.5 ? delta_w[i][j] : 0.0);

                    }

                }
            }
        }

    }

    // Post-synaptic events
    void post_event() {


    }

    // Accessors for default attributes
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }

    // Variable/Parameter access methods

    std::vector<std::vector<double>> get_local_attribute_all(std::string name) {

        if ( name.compare("trace") == 0 ) {
            return get_matrix_variable_all<double>(trace);
        }

        if ( name.compare("delta_w") == 0 ) {
            return get_matrix_variable_all<double>(delta_w);
        }

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable_all<double>(w);
        }


        // should not happen
        std::cerr << "ProjStruct5::get_local_attribute_all: " << name << " not found" << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<double> get_local_attribute_row(std::string name, int rk_post) {

        if ( name.compare("trace") == 0 ) {
            return get_matrix_variable_row<double>(trace, rk_post);
        }

        if ( name.compare("delta_w") == 0 ) {
            return get_matrix_variable_row<double>(delta_w, rk_post);
        }

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable_row<double>(w, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct5::get_local_attribute_row: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute(std::string name, int rk_post, int rk_pre) {

        if ( name.compare("trace") == 0 ) {
            return get_matrix_variable<double>(trace, rk_post, rk_pre);
        }

        if ( name.compare("delta_w") == 0 ) {
            return get_matrix_variable<double>(delta_w, rk_post, rk_pre);
        }

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable<double>(w, rk_post, rk_pre);
        }


        // should not happen
        std::cerr << "ProjStruct5::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all(std::string name, std::vector<std::vector<double>> value) {

        if ( name.compare("trace") == 0 ) {
            update_matrix_variable_all<double>(trace, value);

        }

        if ( name.compare("delta_w") == 0 ) {
            update_matrix_variable_all<double>(delta_w, value);

        }

        if ( name.compare("w") == 0 ) {
            update_matrix_variable_all<double>(w, value);

        }

    }

    void set_local_attribute_row(std::string name, int rk_post, std::vector<double> value) {

        if ( name.compare("trace") == 0 ) {
            update_matrix_variable_row<double>(trace, rk_post, value);

        }

        if ( name.compare("delta_w") == 0 ) {
            update_matrix_variable_row<double>(delta_w, rk_post, value);

        }

        if ( name.compare("w") == 0 ) {
            update_matrix_variable_row<double>(w, rk_post, value);

        }

    }

    void set_local_attribute(std::string name, int rk_post, int rk_pre, double value) {

        if ( name.compare("trace") == 0 ) {
            update_matrix_variable<double>(trace, rk_post, rk_pre, value);

        }

        if ( name.compare("delta_w") == 0 ) {
            update_matrix_variable<double>(delta_w, rk_post, rk_pre, value);

        }

        if ( name.compare("w") == 0 ) {
            update_matrix_variable<double>(w, rk_post, rk_pre, value);

        }

    }

    std::vector<double> get_semiglobal_attribute_all(std::string name) {


        // should not happen
        std::cerr << "ProjStruct5::get_semiglobal_attribute_all: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_semiglobal_attribute(std::string name, int rk_post) {


        // should not happen
        std::cerr << "ProjStruct5::get_semiglobal_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_semiglobal_attribute_all(std::string name, std::vector<double> value) {

    }

    void set_semiglobal_attribute(std::string name, int rk_post, double value) {

    }

    double get_global_attribute(std::string name) {

        if ( name.compare("eta") == 0 ) {
            return eta;
        }

        if ( name.compare("learning_phase") == 0 ) {
            return learning_phase;
        }

        if ( name.compare("error") == 0 ) {
            return error;
        }

        if ( name.compare("mean_error") == 0 ) {
            return mean_error;
        }

        if ( name.compare("max_weight_change") == 0 ) {
            return max_weight_change;
        }


        // should not happen
        std::cerr << "ProjStruct5::get_global_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute(std::string name, double value) {

        if ( name.compare("eta") == 0 ) {
            eta = value;

        }

        if ( name.compare("learning_phase") == 0 ) {
            learning_phase = value;

        }

        if ( name.compare("error") == 0 ) {
            error = value;

        }

        if ( name.compare("mean_error") == 0 ) {
            mean_error = value;

        }

        if ( name.compare("max_weight_change") == 0 ) {
            max_weight_change = value;

        }

    }


    // Access additional


    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<LILMatrix<int>*>(this)->size_in_bytes();
        // local variable trace
        size_in_bytes += sizeof(double) * trace.capacity();
        for(auto it = trace.begin(); it != trace.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // local variable delta_w
        size_in_bytes += sizeof(double) * delta_w.capacity();
        for(auto it = delta_w.begin(); it != delta_w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // local variable w
        size_in_bytes += sizeof(double) * w.capacity();
        for(auto it = w.begin(); it != w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // global parameter eta
        size_in_bytes += sizeof(double);	// eta
        // global parameter learning_phase
        size_in_bytes += sizeof(double);	// learning_phase
        // global parameter error
        size_in_bytes += sizeof(double);	// error
        // global parameter mean_error
        size_in_bytes += sizeof(double);	// mean_error
        // global parameter max_weight_change
        size_in_bytes += sizeof(double);	// max_weight_change

        return size_in_bytes;
    }

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct5::clear()" << std::endl;
    #endif

    }
};

