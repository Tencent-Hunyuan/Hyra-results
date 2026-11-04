// Spec draft-3 section 16 VERIFY examples, independent of certificate decisions.
void draft3_verify_regressions(const Model& model) {
    require(model.n==5&&model.m==4&&model.cap==4,"expected five-qubit cap-4 test model");
    State state(model); state.root(0);
    auto set=[&](std::initializer_list<int> labels) {
        int q=0; for(int a:labels) state.change(q++,a);
    };
    int accepted=0,rejected=0;
    auto pass=[&](auto action) { action(); ++accepted; };
    auto fail=[&](auto action) {
        try {action();} catch(const std::exception&) {++rejected;return;}
        throw std::runtime_error("invalid draft-3 proof accepted");
    };
    // An empty modified domain is determined with contribution 0.
    set({0,1,0,4,1});
    v2verify::Local local(state,{0},1);
    cov2::Action extra{3,1};
    pass([&]{local.replay({{cov2::Step::odd,3,-1}},&extra);});
    pass([&]{local.replay({{cov2::Step::wiped,-1,0}},&extra);});
    fail([&]{local.replay({{cov2::Step::odd,2,-1}},&extra);}); // target 0, no contradiction
    fail([&]{local.replay({{cov2::Step::elim,3,0},{cov2::Step::wiped,-1,0}},&extra);}); // no removal
    fail([&]{local.replay({{cov2::Step::wiped,-1,0},{cov2::Step::odd,3,-1}},&extra);}); // two ENDs
    v2decide::System sys(state,{0});
    auto u=v2decide::unit(sys,sys.flips(extra),extra.q);
    require(u.refuted()&&u.first[0].empty()&&u.values[0].empty()&&u.wiped==std::vector<int>{0},"excluded values still have UNIT records");
    pass([&]{local.replay(v2decide::unit_derivation(sys,u),&extra);});

    // Partial exclusion preserves the surviving values' canonical order.
    set({1,4,4,4,4});
    v2decide::System partial(state,{3});
    auto partial_u=v2decide::unit(partial,partial.flips(extra),extra.q);
    require(partial_u.values[0]==std::vector<size_t>({0,1,4,5})&&partial_u.first[0].size()==4,"modified UNIT domain order mismatch");

    // This node has completion X_0 Z_4 (total Pauli XIXZZ, weight four).
    // Full killer list is [g3, g2]. A shortened [g2] would miss the real extra X_0.
    set({4,0,1,2,4});
    v2verify::Local anchor(state,{1},1);
    require(anchor.domains[0]==std::vector<cov2::Action>({{4,2},{4,3}}),"unexpected canonical domain");
    pass([&]{anchor.kill_list(0,{3,2},nullptr);});
    cov2::Action real_extra{0,1};
    for(const auto& ks:std::vector<std::vector<int>>{{},{2},{3,2,2}}) {
        // The exact same checker routine guards WIPED anchors and KILL reasons.
        fail([&]{anchor.kill_list(0,ks,nullptr);});
        fail([&]{anchor.kill_list(0,ks,&real_extra);});
    }
    v2decide::System anchor_sys(state,{1});
    auto original=v2decide::unit(anchor_sys);
    require(original.refuted()&&original.first[0]==std::vector<int>({3,2}),"unexpected anchor killers");
    cov2::Justification bad;bad.packing={1};bad.steps=v2decide::unit_derivation(anchor_sys,original);
    bad.anchor=0;bad.killers={2};
    fail([&]{v2verify::verify(state,6,bad);});
    std::cout<<"{\"accepted\":"<<accepted<<",\"rejected\":"<<rejected<<",\"modified_empty_records\":true,\"modified_order\":true}\n";
}
