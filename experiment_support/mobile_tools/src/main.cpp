// Business operations ported from Sierra tau-bench (MIT), see vendor/TAU_BENCH_LICENSE.
// WFGen JSON ABI and transactional file-state adapter for Android and POSIX hosts.
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <sys/file.h>
#include <unistd.h>
using J = nlohmann::json;
using Clock = std::chrono::steady_clock;
using S = std::string;
static void need(bool c, const S &m) {
    if (!c)
        throw std::runtime_error("Error: " + m);
}
static S str(const J &j, const S &k) { return j.at(k).get<S>(); }
static S lower(S s) {
    for (auto &c : s)
        c = std::tolower(static_cast<unsigned char>(c));
    return s;
}
static double money(double v) { return std::nearbyint(v * 100.0) / 100.0; }
static J &obj(J &d, const S &table, const S &id, const S &err) {
    need(d.at(table).contains(id), err);
    return d.at(table).at(id);
}
static J address(const J &p) {
    J a = J::object();
    for (auto k : {"address1", "address2", "city", "state", "country", "zip"})
        a[k] = str(p, k);
    return a;
}
static J &payment(J &d, J &item, const S &id) {
    auto &methods = d.at("users").at(str(item, "user_id")).at("payment_methods");
    need(methods.contains(id), "payment method not found");
    return methods.at(id);
}
static std::vector<S> ids(const J &p, const S &key) { return p.at(key).get<std::vector<S>>(); }
static J &item(J &order, const S &id) {
    for (auto &x : order.at("items"))
        if (str(x, "item_id") == id)
            return x;
    throw std::runtime_error("Error: " + id + " not found");
}
static void check_items(J &order, const std::vector<S> &wanted, bool returning = false) {
    std::vector<S> all;
    for (auto &x : order.at("items"))
        all.push_back(str(x, "item_id"));
    for (auto &id : wanted)
        need(std::count(wanted.begin(), wanted.end(), id) <= std::count(all.begin(), all.end(), id),
             returning ? "some item not found" : id + " not found");
}
static J invoke(J &d, const S &domain, const S &tool, const J &p) {
    if (tool == "find_user_by_contact" || tool == "find_user_by_name") {
        for (auto it = d.at("users").begin(); it != d.at("users").end(); ++it) {
            auto &u = it.value();
            bool found =
                tool == "find_user_by_contact"
                    ? lower(str(u, "email")) == lower(str(p, "email"))
                    : lower(str(u.at("name"), "first_name")) == lower(str(p, "first_name")) &&
                          lower(str(u.at("name"), "last_name")) == lower(str(p, "last_name")) &&
                          str(u.at("address"), "zip") == str(p, "zip");
            if (found)
                return it.key();
        }
        need(false, "user not found");
    }
    if (tool == "get_user_details")
        return obj(d, "users", str(p, "user_id"), "user not found");
    if (tool == "get_order_details")
        return obj(d, domain == "airline" ? "reservations" : "orders", str(p, "order_id"),
                   domain == "airline" ? "user not found" : "order not found");
    if (tool == "get_product_details")
        return obj(d, "products", str(p, "product_id"), "product not found");
    if (tool == "modify_user_address") {
        auto &u = obj(d, "users", str(p, "user_id"), "user not found");
        u["address"] = address(p);
        return u;
    }
    if (domain == "airline") {
        S rid = tool == "cancel_order" ? str(p, "order_id") : str(p, "reservation_id");
        auto &r = obj(d, "reservations", rid, "reservation not found");
        if (tool == "cancel_order") {
            J refunds = J::array();
            for (auto &x : r.at("payment_history"))
                refunds.push_back({{"payment_id", x.at("payment_id")},
                                   {"amount", -x.at("amount").get<double>()}});
            for (auto &x : refunds)
                r["payment_history"].push_back(x);
            r["status"] = "cancelled";
            return r;
        }
        if (tool == "update_reservation_baggages") {
            int total = p.at("total_baggages").get<int>(),
                nonfree = p.at("nonfree_baggages").get<int>();
            need(total >= 0 && nonfree >= 0 && nonfree <= total, "invalid baggage counts");
            double cost = 50 * std::max(0, nonfree - r.at("nonfree_baggages").get<int>());
            S pay = str(p, "payment_id");
            auto &pm = payment(d, r, pay);
            S src = str(pm, "source");
            need(src != "certificate", "certificate cannot be used to update reservation");
            need(src != "gift_card" || pm.at("amount").get<double>() >= cost,
                 "gift card balance is not enough");
            r["total_baggages"] = total;
            r["nonfree_baggages"] = nonfree;
            if (src == "gift_card")
                pm["amount"] = pm.at("amount").get<double>() - cost;
            if (cost != 0)
                r["payment_history"].push_back({{"payment_id", pay}, {"amount", cost}});
            return r;
        }
        if (tool == "update_reservation_flights") {
            J flights = p.at("flights");
            need(flights.is_array() && !flights.empty(), "flights must be nonempty");
            S cabin = str(p, "cabin");
            need(cabin == "economy" || cabin == "basic_economy" || cabin == "business",
                 "invalid cabin");
            double cost = 0;
            auto passengers = r.at("passengers").size();
            for (auto &f : flights) {
                bool exists = false;
                for (auto &old : r.at("flights"))
                    if (f.at("flight_number") == old.at("flight_number") &&
                        f.at("date") == old.at("date") && cabin == str(r, "cabin")) {
                        for (auto k : {"price", "origin", "destination"})
                            f[k] = old.at(k);
                        cost += old.at("price").get<double>() * passengers;
                        exists = true;
                        break;
                    }
                if (exists)
                    continue;
                S num = str(f, "flight_number"), date = str(f, "date");
                auto &fd = obj(d, "flights", num, "flight " + num + " not found");
                need(fd.at("dates").contains(date), "flight " + num + " not found on date " + date);
                auto &day = fd.at("dates").at(date);
                need(day.at("status") == "available",
                     "flight " + num + " not available on date " + date);
                need(day.at("available_seats").at(cabin).get<int>() >= static_cast<int>(passengers),
                     "not enough seats on flight " + num);
                f["price"] = day.at("prices").at(cabin);
                f["origin"] = fd.at("origin");
                f["destination"] = fd.at("destination");
                cost += f.at("price").get<double>() * passengers;
            }
            double old_total = 0;
            for (auto &f : r.at("flights"))
                old_total += f.at("price").get<double>();
            cost -= old_total * passengers;
            S pay = str(p, "payment_id");
            auto &pm = payment(d, r, pay);
            S src = str(pm, "source");
            need(src != "certificate", "certificate cannot be used to update reservation");
            need(src != "gift_card" || pm.at("amount").get<double>() >= cost,
                 "gift card balance is not enough");
            if (src == "gift_card")
                pm["amount"] = pm.at("amount").get<double>() - cost;
            r["flights"] = flights;
            if (cost != 0)
                r["payment_history"].push_back({{"payment_id", pay}, {"amount", cost}});
            // Preserve the pinned reference behavior: no cabin or flight database update.
            return r;
        }
        need(false, "unsupported airline tool");
    }
    auto &o = obj(d, "orders", str(p, "order_id"), "order not found");
    if (tool == "cancel_order") {
        need(o.at("status") == "pending", "non-pending order cannot be cancelled");
        S reason = str(p, "reason");
        need(reason == "no longer needed" || reason == "ordered by mistake", "invalid reason");
        J refunds = J::array();
        for (auto &x : o.at("payment_history")) {
            S pay = str(x, "payment_method_id");
            double amt = x.at("amount");
            refunds.push_back(
                {{"transaction_type", "refund"}, {"amount", amt}, {"payment_method_id", pay}});
            if (pay.find("gift_card") != S::npos) {
                auto &pm = payment(d, o, pay);
                pm["balance"] = money(pm.at("balance").get<double>() + amt);
            }
        }
        o["status"] = "cancelled";
        o["cancel_reason"] = reason;
        for (auto &x : refunds)
            o["payment_history"].push_back(x);
        return o;
    }
    if (tool == "return_delivered_order_items") {
        need(o.at("status") == "delivered", "non-delivered order cannot be returned");
        S pay = str(p, "payment_method_id");
        payment(d, o, pay);
        need(pay.find("gift_card") != S::npos ||
                 pay == str(o.at("payment_history").at(0), "payment_method_id"),
             "payment method should be either the original payment method or a gift card");
        auto old = ids(p, "item_ids");
        check_items(o, old, true);
        std::sort(old.begin(), old.end());
        o["status"] = "return requested";
        o["return_items"] = old;
        o["return_payment_method_id"] = pay;
        return o;
    }
    bool exchanging = tool == "exchange_delivered_order_items";
    if (exchanging || tool == "modify_pending_order_items") {
        need(o.at("status") == (exchanging ? "delivered" : "pending"),
             exchanging ? "non-delivered order cannot be exchanged"
                        : "non-pending order cannot be modified");
        auto old = ids(p, "item_ids"), fresh = ids(p, "new_item_ids");
        check_items(o, old);
        need(old.size() == fresh.size(), "the number of items to be exchanged should match");
        double diff = 0;
        for (size_t i = 0; i < old.size(); ++i) {
            auto &x = item(o, old[i]);
            auto &variants = d.at("products").at(str(x, "product_id")).at("variants");
            need(variants.contains(fresh[i]) && variants.at(fresh[i]).at("available").get<bool>(),
                 "new item " + fresh[i] + " not found or available");
            diff += variants.at(fresh[i]).at("price").get<double>() - x.at("price").get<double>();
        }
        if (exchanging)
            diff = money(diff);
        S pay = str(p, "payment_method_id");
        auto &pm = payment(d, o, pay);
        need(pm.at("source") != "gift_card" || pm.at("balance").get<double>() >= diff,
             exchanging ? "insufficient gift card balance to pay for the price difference"
                        : "insufficient gift card balance to pay for the new item");
        if (exchanging) {
            std::sort(old.begin(), old.end());
            std::sort(fresh.begin(), fresh.end());
            o["status"] = "exchange requested";
            o["exchange_items"] = old;
            o["exchange_new_items"] = fresh;
            o["exchange_payment_method_id"] = pay;
            o["exchange_price_difference"] = diff;
        } else {
            o["payment_history"].push_back({{"transaction_type", diff > 0 ? "payment" : "refund"},
                                            {"amount", std::abs(diff)},
                                            {"payment_method_id", pay}});
            if (pm.at("source") == "gift_card")
                pm["balance"] = money(pm.at("balance").get<double>() - diff);
            for (size_t i = 0; i < old.size(); ++i) {
                auto &x = item(o, old[i]);
                auto v = d.at("products").at(str(x, "product_id")).at("variants").at(fresh[i]);
                x["item_id"] = fresh[i];
                x["price"] = v.at("price");
                x["options"] = v.at("options");
            }
            o["status"] = "pending (item modified)";
        }
        return o;
    }
    need(o.at("status") == "pending", "non-pending order cannot be modified");
    if (tool == "modify_pending_order_address") {
        o["address"] = address(p);
        return o;
    }
    if (tool == "modify_pending_order_payment") {
        S pay = str(p, "payment_method_id");
        auto &pm = payment(d, o, pay);
        auto &h = o.at("payment_history");
        need(h.size() == 1 && h.at(0).at("transaction_type") == "payment",
             "there should be exactly one payment for a pending order");
        S previous = str(h.at(0), "payment_method_id");
        need(previous != pay, "the new payment method should be different from the current one");
        double amount = h.at(0).at("amount");
        need(pm.at("source") != "gift_card" || pm.at("balance").get<double>() >= amount,
             "insufficient gift card balance to pay for the order");
        h.push_back(
            {{"transaction_type", "payment"}, {"amount", amount}, {"payment_method_id", pay}});
        h.push_back(
            {{"transaction_type", "refund"}, {"amount", amount}, {"payment_method_id", previous}});
        if (pm.at("source") == "gift_card")
            pm["balance"] = money(pm.at("balance").get<double>() - amount);
        if (previous.find("gift_card") != S::npos) {
            auto &before = payment(d, o, previous);
            before["balance"] = money(before.at("balance").get<double>() + amount);
        }
        return o;
    }
    need(false, "unsupported tool");
    return nullptr;
}
static const std::set<S> readers = {"find_user_by_contact", "find_user_by_name", "get_user_details",
                                    "get_order_details", "get_product_details"};
static const std::set<S> writers = {"cancel_order",
                                    "return_delivered_order_items",
                                    "exchange_delivered_order_items",
                                    "modify_pending_order_items",
                                    "modify_pending_order_address",
                                    "modify_pending_order_payment",
                                    "modify_user_address",
                                    "update_reservation_flights",
                                    "update_reservation_baggages"};
static J artifacts(const S &tool, const J &result, const J &data, const J &p) {
    if (tool == "find_user_by_contact" || tool == "find_user_by_name" ||
        tool == "get_user_details") {
        S id = tool == "get_user_details" ? str(p, "user_id") : result.get<S>();
        J u = data.at("users").at(id);
        u["user_id"] = id;
        return {{"user", u}, {"user_id", id}};
    }
    if (tool == "get_order_details")
        return {{"order", result}, {"order_id", str(p, "order_id")}, {"reservation", result}};
    if (tool == "get_product_details")
        return {{"product", result}};
    static const std::map<S, S> names = {
        {"cancel_order", "cancellation_result"},
        {"return_delivered_order_items", "return_result"},
        {"exchange_delivered_order_items", "exchange_result"},
        {"modify_pending_order_items", "order_modification_result"},
        {"modify_pending_order_address", "address_modification_result"},
        {"modify_pending_order_payment", "payment_modification_result"},
        {"modify_user_address", "user_address_result"},
        {"update_reservation_flights", "reservation_update_result"},
        {"update_reservation_baggages", "baggage_update_result"}};
    return {{names.at(tool), result}};
}
struct Lock {
    int fd;
    Lock(const S &path, bool write) : fd(open((path + ".lock").c_str(), O_CREAT | O_RDWR, 0600)) {
        need(fd >= 0, "cannot open state lock");
        if (flock(fd, write ? LOCK_EX : LOCK_SH) != 0) {
            close(fd);
            throw std::runtime_error("Error: cannot acquire state lock");
        }
    }
    ~Lock() {
        flock(fd, LOCK_UN);
        close(fd);
    }
};
static void atomic_save(const S &path, const J &state) {
    S tmp = path + ".tmp.XXXXXX";
    std::vector<char> name(tmp.begin(), tmp.end());
    name.push_back(0);
    int fd = mkstemp(name.data());
    need(fd >= 0, "cannot create temporary state");
    S bytes = state.dump();
    size_t at = 0;
    bool ok = true;
    while (at < bytes.size()) {
        auto n = write(fd, bytes.data() + at, bytes.size() - at);
        if (n <= 0) {
            ok = false;
            break;
        }
        at += static_cast<size_t>(n);
    }
    if (fsync(fd) != 0)
        ok = false;
    if (close(fd) != 0)
        ok = false;
    if (!ok || rename(name.data(), path.c_str()) != 0) {
        unlink(name.data());
        throw std::runtime_error("Error: cannot commit state");
    }
}
int main(int argc, char **argv) {
    auto begin = Clock::now();
    J response;
    S state_path, tool;
    try {
        if (argc == 2 && S(argv[1]) == "--version") {
            std::cout << "wfgen-tools 1.0\n";
            return 0;
        }
        if (argc < 4)
            throw std::runtime_error("Usage: wfgen-tools TOOL --state STATE.json < request.json");
        tool = argv[1];
        for (int i = 2; i < argc; i++) {
            if (S(argv[i]) == "--state" && i + 1 < argc)
                state_path = argv[++i];
            else
                throw std::runtime_error("Unknown argument");
        }
        need(!state_path.empty(), "state path required");
        need(readers.count(tool) || writers.count(tool), "unknown tool: " + tool);
        J request;
        std::cin >> request;
        need(request.is_object(), "request must be object");
        if (request.contains("entrypoint"))
            need(request.at("entrypoint") == tool, "entrypoint mismatch");
        J params = request.value("parameters", J::object());
        need(params.is_object(), "parameters must be object");
        for (auto k : {"total_baggages", "nonfree_baggages"})
            if (params.contains(k) && params.at(k).is_string()) {
                S v = params.at(k);
                size_t used = 0;
                int n = std::stoi(v, &used);
                need(used == v.size(), "invalid integer");
                params[k] = n;
            }
        Lock lock(state_path, writers.count(tool));
        std::ifstream input(state_path);
        need(input.good(), "state file not found");
        J state;
        input >> state;
        input.close();
        S domain = str(state, "domain");
        need(domain == "retail" || domain == "airline", "unknown state domain");
        J before = state;
        auto function_begin = Clock::now();
        J value = invoke(state.at("data"), domain, tool, params);
        auto function_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - function_begin).count();
        J output = artifacts(tool, value, state.at("data"), params);
        bool changed = state != before;
        if (changed) {
            need(writers.count(tool), "read tool changed state");
            atomic_save(state_path, state);
        }
        response = {
            {"success", true},
            {"artifacts", output},
            {"raw",
             {{"result", value}, {"state_changed", changed}, {"tool_compute_ms", function_ms}}}};
    } catch (const std::exception &e) {
        response = {{"success", false},
                    {"artifacts", J::object()},
                    {"error", e.what()},
                    {"raw", {{"state_changed", false}}}};
    }
    response["wall_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    response["raw"]["mode"] = "phone_native_json_cli";
    response["raw"]["pid"] = getpid();
    response["raw"]["state_path"] = state_path;
#ifdef __ANDROID__
    response["raw"]["platform"] = "android";
#else
    response["raw"]["platform"] = "host_validation";
#endif
    std::cout << response.dump() << "\n";
    return response.at("success").get<bool>() ? 0 : 1;
}
