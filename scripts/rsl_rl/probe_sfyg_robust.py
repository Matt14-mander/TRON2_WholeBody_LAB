"""Read back physical randomization/force buffers. Not locomotion acceptance."""
import json
import torch


def force_buffers(robot):
    composer = getattr(robot, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "composed_force_as_torch"):
        return composer.composed_force_as_torch, composer.composed_torque_as_torch
    force = getattr(robot, "_external_force_b", None)
    return (force, getattr(robot, "_external_torque_b", None))


def probe(env, steps, output, app):
    raw = env.unwrapped
    s = raw._sfyg_robust
    report = {"kind": "payload_robustness_physics_probe", "is_walking_acceptance": False,
        "steps": 0, "interrupted": False, "peak_force_n": 0., "peak_vertical_force_n": 0.,
        "peak_buffer_error_n": 0., "peak_nonpayload_force_n": 0., "peak_torque_nm": 0.,
        "force_readback_available": False, "reset_force_error_n": 0.}
    low, high = float("inf"), float("-inf")
    selected = (~s.nominal).nonzero(as_tuple=False).flatten()
    selected = int(selected[0]) if len(selected) else 0
    with (output / "probe_steps.jsonl").open("w",encoding="utf-8") as stream:
        with torch.no_grad():
            for step in range(steps):
                if not app.is_running():
                    report["interrupted"] = True
                    break
                _,_,_,info = env.step(torch.zeros(raw.num_envs,10,device=raw.device))
                state = info["reset_state"]
                force = state["payload_force_world"]
                report["steps"] += 1
                report["peak_force_n"] = max(report["peak_force_n"],float(force.norm(dim=-1).max()))
                report["peak_vertical_force_n"] = max(report["peak_vertical_force_n"],float(force[:,2].abs().max()))
                low,high = min(low,float(state["payload_mass"].min())),max(high,float(state["payload_mass"].max()))
                buffer,torque = force_buffers(s.robot)
                if buffer is not None:
                    report["force_readback_available"] = True
                    report["peak_buffer_error_n"] = max(report["peak_buffer_error_n"],
                        float((buffer[:,s.payload:s.payload+1] - s.last_local).abs().max()))
                    rest = torch.cat((buffer[:,:s.payload],buffer[:,s.payload+1:]),1)
                    report["peak_nonpayload_force_n"] = max(report["peak_nonpayload_force_n"],float(rest.abs().max()))
                    report["peak_torque_nm"] = max(report["peak_torque_nm"],float(torque.abs().max()))
                stream.write(json.dumps({"time_s":(step+1)*raw.step_dt,
                    **{k:v[selected].tolist() for k,v in state.items()}})+"\n")
                if step == steps//2:
                    raw.reset()
                    buffer,_ = force_buffers(s.robot)
                    if buffer is not None:
                        report["reset_force_error_n"] = float(buffer.abs().max())
    actual = s.robot.root_physx_view.get_masses().to(raw.device)
    inertias = s.robot.root_physx_view.get_inertias().cpu()
    expected = s.robot.data.default_inertia.cpu()[:,s.payload] * (
        actual[:,s.payload].cpu()/s.robot.data.default_mass.cpu()[:,s.payload])[:,None]
    materials = s.robot.root_physx_view.get_material_properties()
    expected_materials = s.materials.clone()
    expected_materials[...,:2] *= s.frictions.cpu()[:,None,None]
    report.update(mass_range_observed=[low,high],
        mass_cache_error_kg=float((actual-s.masses).abs().max()),
        inertia_error=float((inertias[:,s.payload]-expected).abs().max()),
        material_error=float((materials-expected_materials).abs().max()),
        nominal_count=int(s.nominal.sum()),
        nominal_mass_error_kg=float((actual[s.nominal,s.payload]-s.cfg.nominal_mass_kg).abs().max()) if bool(s.nominal.any()) else 0.,
        nominal_force_error_n=float(s.schedule.current[s.nominal].abs().max()) if bool(s.nominal.any()) else 0.)
    report["passed_physics_checks"] = bool(not report["interrupted"] and report["steps"] == steps
        and low >= min(s.cfg.mass_range[0],s.cfg.nominal_mass_kg)-1e-5
        and high <= max(s.cfg.mass_range[1],s.cfg.nominal_mass_kg)+1e-5
        and report["peak_force_n"] <= s.cfg.max_force_n+1e-5
        and (report["peak_force_n"] > 0 or s.cfg.max_force_n == 0 or bool(s.nominal.all()))
        and all(report[k] < 1e-5 for k in ("peak_vertical_force_n","peak_buffer_error_n",
            "peak_nonpayload_force_n","peak_torque_nm","reset_force_error_n","mass_cache_error_kg",
            "inertia_error","material_error","nominal_mass_error_kg","nominal_force_error_n")))
    (output / "physics_probe.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"[ROBUST PROBE] {report}",flush=True)
    if not report["passed_physics_checks"]:
        raise RuntimeError("Payload robustness physics checks failed; inspect physics_probe.json")
