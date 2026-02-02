import json
import matplotlib.pyplot as plt
import os



FIG_SIZE = (7, 3.5)



def get_ax(fig_size=FIG_SIZE):
    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
    return ax

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
def resource_usage_comparison(metrics_data, metric, metric_label, scale=1):
    # Plot
    ax = get_ax()

    files = [
        {"data": metrics_data["wasm_resource_usage.jsonl"], "label": "Standalone Wasm"},
        {"data": metrics_data["docker_resource_usage.jsonl"], "label": "Docker Container"},
        {"data": metrics_data["wasmtime_resource_usage.jsonl"], "label": "Wasm Container"},
    ]

    for file in files:
        x = get_x_from_timestamp(file["data"])
        y = [record[metric] * scale for record in file["data"]]
        ax.plot(x, y, label=file["label"])

    ax.legend()
    plt.ylabel(metric_label)
    plt.xlabel("Time (s)")

    # Show plot
    plt.show()

# metric: "Qual métrica será plotada no gráfico (fps, inference_latency_ms, ...)"
# metric_label: "Nome a ser usado no eixo y do gráfico"

def box_plot(metrics_data, metric, metric_label, multi_client):
    ax = get_ax()

    if multi_client:
        files = [
            {"data": metrics_data["multi_client_k8s_video_metrics.json"], "label": "K8s Scheduler"},
            {"data": metrics_data["multi_client_video_metrics.json"], "label": "Custom Orchestrator"},
        ]
    else:
        files = [
            {"data": metrics_data["wasm_video_metrics.json"], "label": "Standalone Wasm"},
            {"data": metrics_data["docker_video_metrics.json"], "label": "Docker Container"},
            {"data": metrics_data["wasmtime_video_metrics.json"], "label": "Wasm Container"},
        ]
    plot_data = []
    plot_labels = []

    for file in files:
        data = []

        for entry in file["data"]:
            if metric in entry.keys():
                if type(entry[metric]) is dict:
                    data.append(entry[metric]["avg"])
                else:
                    data.append(entry[metric])
    
        plot_data.append(data)
        plot_labels.append(file["label"])

    bp = ax.boxplot(plot_data, labels=plot_labels, showmeans=True, patch_artist=True)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    plt.ylabel(metric_label)
    plt.grid(True)
    plt.show()

def deploy_results(metrics_data):
    ax = get_ax()

    files = [
        {"data": metrics_data["multi_client_k8s_deploy_results.json"],    "label": "K8s Scheduler"},
        {"data": metrics_data["multi_client_deploy_results.json"],    "label": "Custom Orchestrator"},
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


    ax.bar(labels, values)
    ax.set_ylabel("Deploy success rate (%)")

    plt.show()

    

def cold_start(metrics_data, y_label):
    ax = get_ax()

    files = [
        {"data": metrics_data["docker_cold_start.json"],        "label": "Docker Container"},
        {"data": metrics_data["wasm_cold_start.json"],          "label": "Wasm Container"},
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


    ax.bar(labels, values)

    ax.set_ylabel(y_label)

    plt.show()

def deploy_time(metrics_data, y_label):
    ax = get_ax()

    files = [
        {"data": metrics_data["multi_client_k8s_deploy_results.json"],    "label": "K8s Scheduler"},
        {"data": metrics_data["multi_client_deploy_results.json"],    "label": "Custom Orchestrator"},
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
                sum += record["deploy_time"] if "deploy_time" in record.keys() else record["time_to_ready"]

        avg = sum / count

        values.append(avg)
        labels.append(file["label"])


    bars = ax.bar(labels, values)
    ax.set_ylabel(y_label)

    plt.show()

def main():
    plt.rcParams.update({
        'font.size': 12,            
        'axes.titlesize': 12,       
        'axes.labelsize': 12,       
        'legend.fontsize': 9,
        'xtick.labelsize': 12,      
        'ytick.labelsize': 12,       
        'lines.linewidth': 1.8,
    })


    metrics_data = load("metrics/")
    resource_usage_comparison(metrics_data, "memory_bytes", "Memory (MB)", scale=1 / 1_000_000)
    resource_usage_comparison(metrics_data, "cpu_system_pct", "CPU Usage (%)")
    box_plot(metrics_data, "fps_user", "FPS", False)
    box_plot(metrics_data, "inference_latency_ms", "Remote inference (ms)", False)
    box_plot(metrics_data, "missed_frames", "Timeouts per run", True)
    box_plot(metrics_data, "fps_user", "FPS", True)
    box_plot(metrics_data, "inference_latency_ms", "Remote inference (ms)", True)
    deploy_results(metrics_data)
    cold_start(metrics_data, "Cold start time (s)")
    deploy_time(metrics_data, "Deploy time (s)")
    

if __name__ == "__main__":
    main()
