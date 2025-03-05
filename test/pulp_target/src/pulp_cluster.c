#include <pulp_cluster/pulp_cluster.h>

struct pi_device cluster_dev = {0};
struct pi_cluster_task cluster_task = {0};

void pulp_cluster_init() {
    // TODO: Capire se queste parti cambiano
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_FC, 50000000);
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 50000000);
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 50000000);
    pi_time_wait_us(10000);
    struct pi_cluster_conf conf;
    struct pi_cluster_task cluster_task = {0};
    // First open the cluster
    pi_cluster_conf_init(&conf);
    conf.id=0;
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev))
        return;
    // Init task, questi numeri sono per GAP9
    #ifndef TARGET_CHIP_FAMILY_GAP9
    cluster_task.stack_size = 3500;
    #endif
    cluster_task.slave_stack_size = 3400;
}

void pulp_cluster_close() {
    // TODO: Do i need a reset?
    //cluster_dev = {0};
    //cluster_task = {0};
    pi_cluster_close(&cluster_dev);
}
