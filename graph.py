import json
import matplotlib.pyplot as plt
import os







def get_x_from_timestamp(file_data):
    t0 = file_data[0]["timestamp"]
    x = [record["timestamp"] - t0 for record in file_data]
    return x

def load(data_dir):
    files_data = {}

    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        with open(path, "r") as f:
            if filename.endswith(".json"):   
                data = json.load(f)

            if filename.endswith(".jsonl"):
                data = []
                for line in f:
                    data.append(json.loads(line))
            

        files_data[filename] = data
    
    return files_data








# metric: "Qual métrica será plotada no gráfico (memory_bytes, cpu_cores_sum, ...)"
# metric_label: "Nome a ser usado no eixo y do gráfico"
def resource_usage_comparison(metrics_data, metric, metric_label):
    # Plot
    fig, ax = plt.subplots()

    print(metrics_data.keys())

    files = [
        {"data": metrics_data["wasm_resource_usage.jsonl"], "label": "Wasm"},
        {"data": metrics_data["docker_container_resource_usage.jsonl"], "label": "Docker Container"},
        {"data": metrics_data["wasmtime_container_resource_usage.jsonl"], "label": "Wasmtime Container"},
    ]

    for file in files:
        x = get_x_from_timestamp(file["data"])
        y = [record[metric] for record in file["data"]]
        ax.plot(x, y, label=file["label"])

    ax.legend()
    plt.title(f"{metric_label} usage")
    plt.ylabel(metric_label)
    plt.xlabel("Time (s)")

    # Show plot
    plt.show()

# metric: "Qual métrica será plotada no gráfico (fps, inference_latency_ms, ...)"
# metric_label: "Nome a ser usado no eixo y do gráfico"
def box_plot(metrics_data, metric, metric_label, multi_client):
    fig, ax = plt.subplots()

    files = [
        {"data": metrics_data["wasm_metrics.json"], "label": "Wasm"},
        {"data": metrics_data["docker_container_metrics.json"], "label": "Docker Container"},
        {"data": metrics_data["wasmtime_container_metrics.json"], "label": "Wasmtime"},
    ]

    if multi_client:
        files = [
            {"data": metrics_data["multi_client_k8s_metrics.json"], "label": "Multi Client K8s"},
            {"data": metrics_data["multi_client_orchestration_metrics.json"], "label": "Multi Client Orchestration"},
        ]
    else:
        files = [
            {"data": metrics_data["wasm_metrics.json"], "label": "Wasm"},
            {"data": metrics_data["docker_container_metrics.json"], "label": "Docker Container"},
            {"data": metrics_data["wasmtime_container_metrics.json"], "label": "Wasmtime"},
        ]
    plot_data = []
    plot_labels = []

    for file in files:
        avg_fps = [entry[metric]["avg"] for entry in file["data"]]
        plot_data.append(avg_fps)
        plot_labels.append(file["label"])

    bp = ax.boxplot(plot_data, labels=plot_labels, showmeans=True, patch_artist=True)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    plt.ylabel(metric_label)
    plt.title(f"Distribution of Average {metric_label} per Run")
    plt.grid(True)
    plt.show()

def deploy_results(metrics_data):
    fig, ax = plt.subplots()

    files = [
        {"data": metrics_data["multi_client_k8s_scheduler_deploy_results.json"],    "label": "K8s"},
        {"data": metrics_data["multi_client_orchestration_deploy_results.json"],    "label": "Orchestration"},
        {"data": metrics_data["docker_container_deploy_results.json"],              "label": "Docker Container"},
        {"data": metrics_data["wasmtime_container_deploy_results.json"],                          "label": "Wasmtime Container"},
    ]

    labels = []
    values = []

    for file in files:
        total_count = 0
        success_count = 0

        for record in file["data"]:
            total_count += 1

            if record["status"] == "success":
                success_count += 1

        success_rate = int(success_count / total_count * 100)
        values.append(success_rate)
        labels.append(file["label"])


    bars = ax.bar(labels, values)
    ax.bar_label(bars, padding=3)

    ax.set_ylabel("Success rate (%)")
    ax.set_title("Deploy success rate")

    plt.show()

    

def cold_start(metrics_data):
    fig, ax = plt.subplots()

    files = [
        #{"data": metrics_data["multi_client_k8s_scheduler_deploy_results.json"],    "label": "K8s"},
        #{"data": metrics_data["multi_client_orchestration_deploy_results.json"],    "label": "Orchestration"},
        {"data": metrics_data["docker_container_deploy_results.json"],                        "label": "Docker"},
        {"data": metrics_data["wasmtime_container_deploy_results.json"],                      "label": "Wasmtime Container"},
    ]

    labels = []
    values = []

    sum = 0
    count = 0

    for file in files:
        for record in file["data"]:
            count += 1

            if record["status"] == "success":
                count +=1
                sum += record["cold_start_time"]

        avg = sum / count

        values.append(avg)
        labels.append(file["label"])


    bars = ax.bar(labels, values)
    ax.bar_label(bars, padding=3)

    ax.set_ylabel("Time (s)")
    ax.set_title("Avg cold start time")

    plt.show()

def deploy_time(metrics_data):
    fig, ax = plt.subplots()

    files = [
        {"data": metrics_data["multi_client_k8s_scheduler_deploy_results.json"],    "label": "K8s"},
        {"data": metrics_data["multi_client_orchestration_deploy_results.json"],    "label": "Orchestration"},
        #{"data": metrics_data["docker_container_deploy_results.json"],                        "label": "Docker"},
        #{"data": metrics_data["wasmtime_container_deploy_results.json"],                          "label": "Wasmtime"},
    ]

    labels = []
    values = []

    sum = 0
    count = 0

    for file in files:
        for record in file["data"]:
            count += 1

            if record["status"] == "success":
                count +=1
                sum += record["deploy_time"]

        avg = sum / count

        values.append(avg)
        labels.append(file["label"])


    bars = ax.bar(labels, values)
    ax.bar_label(bars, padding=3)

    ax.set_ylabel("Time (s)")
    ax.set_title("Avg deploy time")

    plt.show()

def main():
    metrics_data = load("metrics/")
    #resource_usage_comparison(metrics_data, "memory_bytes", "Memory (B)")
    #resource_usage_comparison(metrics_data, "cpu_cores_sum", "CPU Cores Sum")
    #resource_usage_comparison(metrics_data, "cpu_system_pct", "CPU System")
    #box_plot(metrics_data, "fps", "FPS", True)
    #box_plot(metrics_data, "inference_latency_ms", "Latency (ms)", True)
    #deploy_results(metrics_data)
    #cold_start(metrics_data)
    #deploy_time(metrics_data)
    

if __name__ == "__main__":
    main()
