
int addRecorder(class Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(class Monitor* recorder);

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    ~Monitor() = default;

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;
    virtual void clear() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;
};

class PopRecorder0 : public Monitor
{
protected:
    PopRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << this << ") instantiated." << std::endl;
    #endif

        this->_sum_exc = std::vector< std::vector< double > >();
        this->record__sum_exc = false; 
        this->_sum_in = std::vector< std::vector< double > >();
        this->record__sum_in = false; 
        this->tau = std::vector< std::vector< double > >();
        this->record_tau = false; 
        this->constant = std::vector< std::vector< double > >();
        this->record_constant = false; 
        this->alpha = std::vector< double >();
        this->record_alpha = false; 
        this->f = std::vector< double >();
        this->record_f = false; 
        this->A = std::vector< std::vector< double > >();
        this->record_A = false; 
        this->perturbation = std::vector< std::vector< double > >();
        this->record_perturbation = false; 
        this->noise = std::vector< std::vector< double > >();
        this->record_noise = false; 
        this->x = std::vector< std::vector< double > >();
        this->record_x = false; 
        this->rprev = std::vector< std::vector< double > >();
        this->record_rprev = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->delta_x = std::vector< std::vector< double > >();
        this->record_delta_x = false; 
        this->x_mean = std::vector< std::vector< double > >();
        this->record_x_mean = false; 
    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder0* get_instance(int id) {
        return static_cast<PopRecorder0*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder0::record()" << std::endl;
    #endif

        if(this->record_tau && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->tau.push_back(pop0.tau);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.tau[this->ranks[i]]);
                }
                this->tau.push_back(tmp);
            }
        }
        if(this->record_constant && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->constant.push_back(pop0.constant);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.constant[this->ranks[i]]);
                }
                this->constant.push_back(tmp);
            }
        }
        if(this->record_alpha && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            this->alpha.push_back(pop0.alpha);
        } 
        if(this->record_f && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            this->f.push_back(pop0.f);
        } 
        if(this->record_A && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->A.push_back(pop0.A);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.A[this->ranks[i]]);
                }
                this->A.push_back(tmp);
            }
        }
        if(this->record_perturbation && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->perturbation.push_back(pop0.perturbation);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.perturbation[this->ranks[i]]);
                }
                this->perturbation.push_back(tmp);
            }
        }
        if(this->record_noise && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->noise.push_back(pop0.noise);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.noise[this->ranks[i]]);
                }
                this->noise.push_back(tmp);
            }
        }
        if(this->record_x && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->x.push_back(pop0.x);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.x[this->ranks[i]]);
                }
                this->x.push_back(tmp);
            }
        }
        if(this->record_rprev && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->rprev.push_back(pop0.rprev);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.rprev[this->ranks[i]]);
                }
                this->rprev.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop0.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_delta_x && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->delta_x.push_back(pop0.delta_x);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.delta_x[this->ranks[i]]);
                }
                this->delta_x.push_back(tmp);
            }
        }
        if(this->record_x_mean && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->x_mean.push_back(pop0.x_mean);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.x_mean[this->ranks[i]]);
                }
                this->x_mean.push_back(tmp);
            }
        }
    }

    void record_targets() {

        if(this->record__sum_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->_sum_exc.push_back(pop0._sum_exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0._sum_exc[this->ranks[i]]);
                }
                this->_sum_exc.push_back(tmp);
            }
        }
        if(this->record__sum_in && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->_sum_in.push_back(pop0._sum_in);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0._sum_in[this->ranks[i]]);
                }
                this->_sum_in.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable tau
        size_in_bytes += sizeof(std::vector<double>) * tau.capacity();
        for(auto it=tau.begin(); it!= tau.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable constant
        size_in_bytes += sizeof(std::vector<double>) * constant.capacity();
        for(auto it=constant.begin(); it!= constant.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // global variable alpha
        size_in_bytes += sizeof(double);
        
        // global variable f
        size_in_bytes += sizeof(double);
        
        // local variable A
        size_in_bytes += sizeof(std::vector<double>) * A.capacity();
        for(auto it=A.begin(); it!= A.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable perturbation
        size_in_bytes += sizeof(std::vector<double>) * perturbation.capacity();
        for(auto it=perturbation.begin(); it!= perturbation.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable noise
        size_in_bytes += sizeof(std::vector<double>) * noise.capacity();
        for(auto it=noise.begin(); it!= noise.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable x
        size_in_bytes += sizeof(std::vector<double>) * x.capacity();
        for(auto it=x.begin(); it!= x.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable rprev
        size_in_bytes += sizeof(std::vector<double>) * rprev.capacity();
        for(auto it=rprev.begin(); it!= rprev.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable delta_x
        size_in_bytes += sizeof(std::vector<double>) * delta_x.capacity();
        for(auto it=delta_x.begin(); it!= delta_x.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable x_mean
        size_in_bytes += sizeof(std::vector<double>) * x_mean.capacity();
        for(auto it=x_mean.begin(); it!= x_mean.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::clear()" << std::endl;
    #endif
        
                for(auto it = this->tau.begin(); it != this->tau.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->tau.clear();
            
                for(auto it = this->constant.begin(); it != this->constant.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->constant.clear();
            
                this->alpha.clear();
            
                this->f.clear();
            
                for(auto it = this->A.begin(); it != this->A.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->A.clear();
            
                for(auto it = this->perturbation.begin(); it != this->perturbation.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->perturbation.clear();
            
                for(auto it = this->noise.begin(); it != this->noise.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->noise.clear();
            
                for(auto it = this->x.begin(); it != this->x.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->x.clear();
            
                for(auto it = this->rprev.begin(); it != this->rprev.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->rprev.clear();
            
                for(auto it = this->r.begin(); it != this->r.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->r.clear();
            
                for(auto it = this->delta_x.begin(); it != this->delta_x.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->delta_x.clear();
            
                for(auto it = this->x_mean.begin(); it != this->x_mean.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->x_mean.clear();
            

        removeRecorder(this);
    }



    // Local variable _sum_exc
    std::vector< std::vector< double > > _sum_exc ;
    bool record__sum_exc ; 
    // Local variable _sum_in
    std::vector< std::vector< double > > _sum_in ;
    bool record__sum_in ; 
    // Local variable tau
    std::vector< std::vector< double > > tau ;
    bool record_tau ; 
    // Local variable constant
    std::vector< std::vector< double > > constant ;
    bool record_constant ; 
    // Global variable alpha
    std::vector< double > alpha ;
    bool record_alpha ; 
    // Global variable f
    std::vector< double > f ;
    bool record_f ; 
    // Local variable A
    std::vector< std::vector< double > > A ;
    bool record_A ; 
    // Local variable perturbation
    std::vector< std::vector< double > > perturbation ;
    bool record_perturbation ; 
    // Local variable noise
    std::vector< std::vector< double > > noise ;
    bool record_noise ; 
    // Local variable x
    std::vector< std::vector< double > > x ;
    bool record_x ; 
    // Local variable rprev
    std::vector< std::vector< double > > rprev ;
    bool record_rprev ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable delta_x
    std::vector< std::vector< double > > delta_x ;
    bool record_delta_x ; 
    // Local variable x_mean
    std::vector< std::vector< double > > x_mean ;
    bool record_x_mean ; 
};

class PopRecorder1 : public Monitor
{
protected:
    PopRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << this << ") instantiated." << std::endl;
    #endif

        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder1(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder1* get_instance(int id) {
        return static_cast<PopRecorder1*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder1::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop1.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder1::clear()" << std::endl;
    #endif
        
                for(auto it = this->r.begin(); it != this->r.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->r.clear();
            

        removeRecorder(this);
    }



    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

class PopRecorder2 : public Monitor
{
protected:
    PopRecorder2(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << this << ") instantiated." << std::endl;
    #endif

        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder2(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder2* get_instance(int id) {
        return static_cast<PopRecorder2*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder2::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop2.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder2::clear()" << std::endl;
    #endif
        
                for(auto it = this->r.begin(); it != this->r.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->r.clear();
            

        removeRecorder(this);
    }



    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

class PopRecorder3 : public Monitor
{
protected:
    PopRecorder3(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << this << ") instantiated." << std::endl;
    #endif

        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder3(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder3* get_instance(int id) {
        return static_cast<PopRecorder3*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder3::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop3.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder3::clear()" << std::endl;
    #endif
        
                for(auto it = this->r.begin(); it != this->r.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->r.clear();
            

        removeRecorder(this);
    }



    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

class PopRecorder4 : public Monitor
{
protected:
    PopRecorder4(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder4 (" << this << ") instantiated." << std::endl;
    #endif

        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder4(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder4 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder4* get_instance(int id) {
        return static_cast<PopRecorder4*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder4::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop4.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder4::clear()" << std::endl;
    #endif
        
                for(auto it = this->r.begin(); it != this->r.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->r.clear();
            

        removeRecorder(this);
    }



    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

class PopRecorder5 : public Monitor
{
protected:
    PopRecorder5(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder5 (" << this << ") instantiated." << std::endl;
    #endif

        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder5(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder5 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder5* get_instance(int id) {
        return static_cast<PopRecorder5*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder5::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop5.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder5::clear()" << std::endl;
    #endif
        
                for(auto it = this->r.begin(); it != this->r.end(); it++) {
                    it->clear();
                    it->shrink_to_fit();
                }
                this->r.clear();
            

        removeRecorder(this);
    }



    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

class ProjRecorder0 : public Monitor
{
protected:
    ProjRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj0.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder0* get_instance(int id) {
        return static_cast<ProjRecorder0*>(getRecorder(id));
    }

    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj0.get_matrix_variable_row(proj0.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor0::clear(): not implemented for openMP paradigm." << std::endl;
    }


    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

class ProjRecorder1 : public Monitor
{
protected:
    ProjRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder1 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj1.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder1(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder1 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder1* get_instance(int id) {
        return static_cast<ProjRecorder1*>(getRecorder(id));
    }

    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj1.get_matrix_variable_row(proj1.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor1::clear(): not implemented for openMP paradigm." << std::endl;
    }


    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

class ProjRecorder2 : public Monitor
{
protected:
    ProjRecorder2(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder2 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj2.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder2(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder2 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder2* get_instance(int id) {
        return static_cast<ProjRecorder2*>(getRecorder(id));
    }

    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj2.get_matrix_variable_row(proj2.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor2::clear(): not implemented for openMP paradigm." << std::endl;
    }


    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

class ProjRecorder3 : public Monitor
{
protected:
    ProjRecorder3(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder3 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj3.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder3(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder3 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder3* get_instance(int id) {
        return static_cast<ProjRecorder3*>(getRecorder(id));
    }

    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj3.get_matrix_variable_row(proj3.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor3::clear(): not implemented for openMP paradigm." << std::endl;
    }


    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

class ProjRecorder4 : public Monitor
{
protected:
    ProjRecorder4(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder4 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj4.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder4(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder4 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder4* get_instance(int id) {
        return static_cast<ProjRecorder4*>(getRecorder(id));
    }

    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj4.get_matrix_variable_row(proj4.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor4::clear(): not implemented for openMP paradigm." << std::endl;
    }


    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

class ProjRecorder5 : public Monitor
{
protected:
    ProjRecorder5(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder5 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj5.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->eta = std::vector< double >();
        this->record_eta = false;

        this->learning_phase = std::vector< double >();
        this->record_learning_phase = false;

        this->error = std::vector< double >();
        this->record_error = false;

        this->mean_error = std::vector< double >();
        this->record_mean_error = false;

        this->max_weight_change = std::vector< double >();
        this->record_max_weight_change = false;

        this->trace = std::vector< std::vector< std::vector< double > > >();
        this->record_trace = false;

        this->delta_w = std::vector< std::vector< std::vector< double > > >();
        this->record_delta_w = false;

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder5(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder5 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder5* get_instance(int id) {
        return static_cast<ProjRecorder5*>(getRecorder(id));
    }

    void record() {

        if(this->record_eta && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            this->eta.push_back(proj5.eta);
        }

        if(this->record_learning_phase && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            this->learning_phase.push_back(proj5.learning_phase);
        }

        if(this->record_error && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            this->error.push_back(proj5.error);
        }

        if(this->record_mean_error && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            this->mean_error.push_back(proj5.mean_error);
        }

        if(this->record_max_weight_change && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            this->max_weight_change.push_back(proj5.max_weight_change);
        }

        if(this->record_trace && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj5.get_matrix_variable_row(proj5.trace, this->indices[i])));
            }
            this->trace.push_back(tmp);
            tmp.clear();
        }

        if(this->record_delta_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj5.get_matrix_variable_row(proj5.delta_w, this->indices[i])));
            }
            this->delta_w.push_back(tmp);
            tmp.clear();
        }

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj5.get_matrix_variable_row(proj5.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor5::clear(): not implemented for openMP paradigm." << std::endl;
    }


    // Global variable eta
    std::vector< double > eta ;
    bool record_eta ;

    // Global variable learning_phase
    std::vector< double > learning_phase ;
    bool record_learning_phase ;

    // Global variable error
    std::vector< double > error ;
    bool record_error ;

    // Global variable mean_error
    std::vector< double > mean_error ;
    bool record_mean_error ;

    // Global variable max_weight_change
    std::vector< double > max_weight_change ;
    bool record_max_weight_change ;

    // Local variable trace
    std::vector< std::vector< std::vector< double > > > trace ;
    bool record_trace ;

    // Local variable delta_w
    std::vector< std::vector< std::vector< double > > > delta_w ;
    bool record_delta_w ;

    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

