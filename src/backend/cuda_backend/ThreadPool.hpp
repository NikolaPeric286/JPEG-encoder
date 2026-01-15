#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <vector>


class ThreadPool{

public:

    ThreadPool(size_t thread_count = std::thread::hardware_concurrency());
    ~ThreadPool();

    void enqueueTask(std::function<void()> task);


private:
    
    std::vector<std::thread> thread_vector;

    std::queue<std::function<void()>> task_queue;

    std::mutex queue_mutex;

    std::condition_variable cv;

    bool stop = false;

};
