import { app } from "../../../scripts/app.js";

const CUSTOM_MODE = "Custom — manual values";
const CUSTOM_WIDGETS = [
    "threshold",
    "start_percent",
    "end_percent",
    "max_consecutive_hits",
    "temporal_guard",
];

function updateCustomWidgets(node) {
    const mode = node.widgets?.find((widget) => widget.name === "mode");
    if (!mode) return;

    const disabled = mode.value !== CUSTOM_MODE;
    for (const name of CUSTOM_WIDGETS) {
        const widget = node.widgets.find((candidate) => candidate.name === name);
        if (!widget) continue;
        widget.disabled = disabled;
        widget.options ??= {};
        widget.options.disabled = disabled;
    }
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MiniMaxH3.FirstBlockCache.CustomWidgets",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ApplyMiniMaxH3FirstBlockCache") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const mode = this.widgets?.find((widget) => widget.name === "mode");
            if (mode) {
                const callback = mode.callback;
                mode.callback = (...args) => {
                    callback?.apply(mode, args);
                    updateCustomWidgets(this);
                };
            }
            queueMicrotask(() => updateCustomWidgets(this));
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            queueMicrotask(() => updateCustomWidgets(this));
            return result;
        };
    },
});
