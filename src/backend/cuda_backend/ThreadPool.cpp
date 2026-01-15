#include "ThreadPool.hpp"



ThreadPool::ThreadPool(size_t thread_count){
    
    for(size_t i = 0; i < thread_count; i++){
        thread_vector.emplace_back([this](){
            while(1){
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex); // locks mutex for safe access


                    cv.wait(lock, [this](){ // sleeps thread unless predicate is true, waits till notified
                        return !task_queue.empty() || stop;
                    });

                    if (stop && task_queue.empty()){ // ends thread if queue is empty and stop is true, must be notified
                        return;
                    }

                    task = std::move(task_queue.front()); // grabs function
                    task_queue.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool(){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }

    cv.notify_all();

    for(auto& thread : thread_vector){
        thread.join();
    }
}

void ThreadPool::enqueueTask(std::function<void()> task){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        task_queue.emplace(std::move(task));
    }
    cv.notify_one();
}