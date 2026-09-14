#include "ui.hpp"
#include "ticker.hpp"

#include <SDL.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace pixels {
namespace {
constexpr int Width = 768, Height = 512;
using Color = std::uint32_t;
constexpr Color Ink = 0xff10252c, Panel = 0xff19343a, Edge = 0xff2b494d;
constexpr Color Cream = 0xffe4e9d1, Muted = 0xff90aaa4, Teal = 0xff64c7b2;
constexpr Color Gold = 0xfff2be64, Red = 0xffe88c77, Dark = 0xff0b1e25;
constexpr std::array<Color, 6> Clans{Gold, 0xfff099ab, 0xff91d3e6, 0xffc9b3eb, 0xffa9d88b, 0xffefad7c};

// The left map panel is 480 by 320 at 16,116. The world's pixel cell shrinks to
// fit the chosen preset, so the map is always fully visible and centered.
struct MapView { int x, y, cell, w, h; };
MapView mapView(const Simulation& sim) {
    constexpr int panelW = 480, panelH = 320;
    const int cell = std::max(1, std::min(5, std::min(panelW / sim.width(), panelH / sim.height())));
    const int w = sim.width() * cell, h = sim.height() * cell;
    return {16 + (panelW - w) / 2, 116 + (panelH - h) / 2, cell, w, h};
}

// Original 5 by 7 bitmap lettering: every visible mark is drawn into the pixel canvas.
std::array<unsigned char, 7> glyph(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
    case 'A': return {14,17,17,31,17,17,17};
    case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14};
    case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31};
    case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,15};
    case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {14,4,4,4,4,4,14};
    case 'J': return {7,2,2,2,18,18,12};
    case 'K': return {17,18,20,24,20,18,17};
    case 'L': return {16,16,16,16,16,16,31};
    case 'M': return {17,27,21,21,17,17,17};
    case 'N': return {17,25,25,21,19,19,17};
    case 'O': return {14,17,17,17,17,17,14};
    case 'P': return {30,17,17,30,16,16,16};
    case 'Q': return {14,17,17,17,21,18,13};
    case 'R': return {30,17,17,30,20,18,17};
    case 'S': return {15,16,16,14,1,1,30};
    case 'T': return {31,4,4,4,4,4,4};
    case 'U': return {17,17,17,17,17,17,14};
    case 'V': return {17,17,17,17,17,10,4};
    case 'W': return {17,17,17,21,21,21,10};
    case 'X': return {17,17,10,4,10,17,17};
    case 'Y': return {17,17,10,4,4,4,4};
    case 'Z': return {31,1,2,4,8,16,31};
    case '0': return {14,17,19,21,25,17,14};
    case '1': return {4,12,4,4,4,4,14};
    case '2': return {14,17,1,2,4,8,31};
    case '3': return {30,1,1,14,1,1,30};
    case '4': return {2,6,10,18,31,2,2};
    case '5': return {31,16,16,30,1,1,30};
    case '6': return {14,16,16,30,17,17,14};
    case '7': return {31,1,2,4,8,8,8};
    case '8': return {14,17,17,14,17,17,14};
    case '9': return {14,17,17,15,1,1,14};
    case '.': return {0,0,0,0,0,12,12};
    case ',': return {0,0,0,0,0,4,8};
    case ':': return {0,12,12,0,12,12,0};
    case ';': return {0,12,12,0,4,4,8};
    case '-': return {0,0,0,31,0,0,0};
    case '+': return {0,4,4,31,4,4,0};
    case '/': return {1,1,2,4,8,16,16};
    case '>': return {16,8,4,2,4,8,16};
    case '<': return {1,2,4,8,4,2,1};
    case '=': return {0,0,31,0,31,0,0};
    case '#': return {10,10,31,10,31,10,10};
    case '%': return {25,25,2,4,8,19,19};
    case '[': return {14,8,8,8,8,8,14};
    case ']': return {14,2,2,2,2,2,14};
    case '(': return {2,4,8,8,8,4,2};
    case ')': return {8,4,2,2,2,4,8};
    case '!': return {4,4,4,4,4,0,4};
    case '?': return {14,17,1,2,4,0,4};
    case '|': return {4,4,4,4,4,4,4};
    case '_': return {0,0,0,0,0,0,31};
    default: return {};
    }
}

struct Canvas {
    std::vector<Color> pixels = std::vector<Color>(Width * Height, Ink);
    void clear(Color c) { std::fill(pixels.begin(), pixels.end(), c); }
    void pixel(int x, int y, Color c) {
        if (x >= 0 && x < Width && y >= 0 && y < Height) pixels[y * Width + x] = c;
    }
    void rect(int x, int y, int w, int h, Color c) {
        for (int yy = std::max(0, y); yy < std::min(Height, y + h); ++yy)
            for (int xx = std::max(0, x); xx < std::min(Width, x + w); ++xx)
                pixels[yy * Width + xx] = c;
    }
    void frame(int x, int y, int w, int h, Color c) {
        rect(x,y,w,1,c); rect(x,y+h-1,w,1,c); rect(x,y,1,h,c); rect(x+w-1,y,1,h,c);
    }
    void line(int x0, int y0, int x1, int y1, Color c) {
        int dx = std::abs(x1-x0), sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1-y0), sy = y0 < y1 ? 1 : -1, error = dx+dy;
        for (;;) {
            pixel(x0,y0,c);
            if (x0 == x1 && y0 == y1) break;
            const int twice = 2*error;
            if (twice >= dy) { error += dy; x0 += sx; }
            if (twice <= dx) { error += dx; y0 += sy; }
        }
    }
    void text(int x, int y, const std::string& value, Color c = Cream, int scale = 1) {
        int cursor = x;
        for (char ch : value) {
            if (ch == '\n') { y += 10*scale; cursor = x; continue; }
            const auto shape = glyph(ch);
            for (int row = 0; row < 7; ++row)
                for (int col = 0; col < 5; ++col)
                    if (shape[row] & (1 << (4-col))) rect(cursor+col*scale,y+row*scale,scale,scale,c);
            cursor += 6*scale;
        }
    }
    void wrap(int x, int y, std::string value, int characters, int lines, Color c = Muted) {
        for (int row = 0; row < lines && !value.empty(); ++row) {
            std::size_t length = std::min(value.size(), static_cast<std::size_t>(characters));
            if (length < value.size()) {
                const auto space = value.rfind(' ',length);
                if (space != std::string::npos && space > 0) length = space;
            }
            std::string part = value.substr(0,length);
            value.erase(0,length);
            while (!value.empty() && value.front() == ' ') value.erase(0,1);
            if (row == lines-1 && !value.empty() && part.size() >= 3) part.replace(part.size()-3,3,"...");
            text(x,y+row*10,part,c);
        }
    }
};

std::string decimal(float value, int digits = 2) {
    std::ostringstream out; out << std::fixed << std::setprecision(digits) << value; return out.str();
}
std::string shortened(std::string value, std::size_t length) {
    if (value.size() > length) value = value.substr(0,length-3)+"...";
    return value;
}
Color blend(Color a, Color b, float mix) {
    mix = std::clamp(mix,0.0f,1.0f);
    Color result = 0xff000000;
    for (int shift : {0,8,16}) {
        const float av = static_cast<float>((a >> shift) & 255);
        const float bv = static_cast<float>((b >> shift) & 255);
        result |= static_cast<Color>(av+(bv-av)*mix) << shift;
    }
    return result;
}
int percent(float value) { return static_cast<int>(std::round(std::clamp(value,0.0f,1.0f)*100)); }
Color clanColor(int clan) { return Clans[static_cast<std::size_t>(std::abs(clan)) % Clans.size()]; }
bool inside(int x, int y, int bx, int by, int bw, int bh) {
    return x >= bx && x < bx+bw && y >= by && y < by+bh;
}

struct View {
    int selectedId = -1;
    int layer = 0;
    bool help = false;
};

const Citizen* selected(const Simulation& sim, const View& view) {
    for (const auto& c : sim.citizens()) if (c.id == view.selectedId) return &c;
    return nullptr;
}
void nextCitizen(const Simulation& sim, View& view) {
    for (const auto& c : sim.citizens()) if (c.alive && c.id > view.selectedId) { view.selectedId = c.id; return; }
    for (const auto& c : sim.citizens()) if (c.alive) { view.selectedId = c.id; return; }
    view.selectedId = -1;
}
void selectAt(const Simulation& sim, View& view, int x, int y) {
    int best = 100000;
    for (const auto& c : sim.citizens()) {
        if (!c.alive) continue;
        const int dx = c.x-x, dy = c.y-y, distance = dx*dx+dy*dy;
        if (distance < best) { best = distance; view.selectedId = c.id; }
    }
}

void drawWorld(Canvas& out, const Simulation& sim, const View& view) {
    const bool winter = sim.seasonName() == "Winter" || sim.seasonName() == "WINTER";
    const MapView map = mapView(sim);
    for (int y = 0; y < sim.height(); ++y) {
        for (int x = 0; x < sim.width(); ++x) {
            const auto& tile = sim.tile(x,y);
            const int px = map.x+x*map.cell, py = map.y+y*map.cell;
            const unsigned noise = static_cast<unsigned>(x*1973+y*9277+(x*y)*13);
            Color base = 0xff477963;
            switch (tile.terrain) {
            case Terrain::Water: base = 0xff285f72; break;
            case Terrain::Sand: base = 0xffb8ad79; break;
            case Terrain::Grass: base = winter ? 0xff829d8b : 0xff527e61; break;
            case Terrain::Forest: base = 0xff375e4d; break;
            case Terrain::Rock: base = 0xff76817b; break;
            }
            base = blend(base, (noise%3 == 0 ? Cream : Dark),0.035f*static_cast<float>(noise%3));
            out.rect(px,py,map.cell,map.cell,base);
            if (tile.terrain == Terrain::Water) {
                if ((noise+sim.tick()/3)%9 == 0) out.rect(px+1,py+2,3,1,0xff407d8c);
                if (noise%11 == 0) out.pixel(px,py,0xff4c8793);
            } else if (tile.terrain == Terrain::Forest && tile.wood > 0.15f) {
                out.rect(px+2,py+2,1,3,0xff786c4c);
                out.rect(px+1,py+1,3,2,0xff245340);
                out.rect(px+2,py,1,1,0xff81a66d);
                out.pixel(px+1,py+1,0xff608854);
            } else if (tile.terrain == Terrain::Rock) {
                out.rect(px+1,py+1,3,3,0xff63736d);
                out.rect(px+1,py+1,2,1,0xff9ba394);
            } else {
                if (tile.traffic > 0.5f) out.rect(px+1,py+1,3,3,blend(base,0xffad976a,std::min(0.7f,tile.traffic*0.08f)));
                if (tile.food > 0.4f && noise%3 == 0) out.pixel(px+1,py+2,0xffa2b96c);
                if (tile.food > 2.0f && noise%4 == 0) out.pixel(px+3,py+3,Gold);
            }
            if (view.layer == 1 && tile.terrain != Terrain::Water)
                out.rect(px+1,py+1,3,3,blend(0xff6e503e,Teal,std::clamp(tile.fertility,0.0f,1.0f)));
            if (view.layer == 3 && sim.territory(x,y) >= 0)
                out.rect(px,py,map.cell,map.cell,blend(base,clanColor(sim.territory(x,y)),0.30f));
            if (tile.structure == Structure::Home) {
                out.rect(px+1,py+2,3,3,0xffead6a3);
                out.rect(px,py+1,5,1,0xffab6552); out.rect(px+1,py,3,1,Red);
                out.pixel(px+2,py+4,0xff493f37);
            } else if (tile.structure == Structure::Farm) {
                out.rect(px,py,5,5,0xff6c633f);
                for (int row = 1; row < 5; row += 2) out.rect(px+1,py+row,3,1,tile.food > 0.5f ? Gold : 0xffb3ab66);
            }
            if (tile.fire > 0) {
                out.rect(px+1,py+1,3,4,Red); out.rect(px+2,py+2,1,3,Gold);
                out.pixel(px+1+static_cast<int>(sim.tick()%3),py,Gold);
            }
        }
    }
    for (const auto& c : sim.citizens()) {
        if (!c.alive) continue;
        const int x = map.x+c.x*map.cell, y = map.y+c.y*map.cell;
        const Color color = view.layer == 1 ? blend(Teal,Red,std::max(c.hunger,c.thirst)) : clanColor(c.clan);
        if (view.layer == 2) out.frame(x-1,y-1,7,7,color);
        out.rect(x+1,y+3,3,2,0xff203c37);
        out.rect(x+1,y+1,2,3,color);
        out.pixel(x+2,y, Cream);
    }
    if (const auto* c = selected(sim,view); c && c->alive) {
        const int x = map.x+c->x*map.cell, y = map.y+c->y*map.cell;
        // Brackets keep the resident visible, even in crowded settlements.
        for (int side : {-1,1}) {
            const int xx = x+2+side*5;
            out.rect(xx,y-2,1,9,Cream);
            out.rect(side < 0 ? xx : xx-1,y-2,2,1,Cream);
            out.rect(side < 0 ? xx : xx-1,y+6,2,1,Cream);
        }
    }
    if (view.layer == 3) {
        // Civilisation borders: a bright line wherever neighbouring land is
        // claimed differently (or not at all), except across water and rock.
        constexpr Color Frontier = 0xfff6f2dc;
        const auto& tiles = sim.tiles();
        for (int y = 0; y < sim.height(); ++y) {
            for (int x = 0; x < sim.width(); ++x) {
                const int cell = y*sim.width()+x;
                const int here = sim.territory(x,y);
                if (x+1 < sim.width() &&
                    tiles[static_cast<std::size_t>(cell)].terrain != Terrain::Water &&
                    tiles[static_cast<std::size_t>(cell+1)].terrain != Terrain::Water &&
                    here != sim.territory(x+1,y))
                    out.rect(map.x+(x+1)*map.cell-1,map.y+y*map.cell,1,map.cell,Frontier);
                if (y+1 < sim.height() &&
                    tiles[static_cast<std::size_t>(cell)].terrain != Terrain::Rock &&
                    tiles[static_cast<std::size_t>(cell+sim.width())].terrain != Terrain::Rock &&
                    here != sim.territory(x,y+1))
                    out.rect(map.x+x*map.cell,map.y+(y+1)*map.cell-1,map.cell,1,Frontier);
            }
        }
    }
    out.frame(map.x-1,map.y-1,map.w+2,map.h+2,Edge);
    if (sim.stats().population == 0) {
        out.rect(116,248,280,52,Dark); out.frame(116,248,280,52,Edge);
        out.text(134,260,"THE LAST LIGHT HAS GONE",Cream);
        out.text(134,279,"THE WORLD CONTINUES AT 5 HZ",Muted);
    }
}

void metric(Canvas& out, int x, const std::string& name, const std::string& value, Color color = Cream) {
    out.rect(x,56,88,35,Panel); out.text(x+8,63,name,Muted); out.text(x+8,76,value,color);
}
void compactMeter(Canvas& out, int x, int y, const std::string& label, float value, Color color) {
    out.text(x,y,label,Muted);
    out.rect(x+42,y+1,44,5,Dark);
    out.rect(x+42,y+1,static_cast<int>(44*std::clamp(value,0.0f,1.0f)),5,color);
}
float sensorMean(const Observation& observation, int first, int count) {
    float total = 0.0f;
    for (int i = first; i < first+count; ++i) total += observation[static_cast<std::size_t>(i)];
    return total / static_cast<float>(count);
}
int activeSensors(const Observation& observation, int first, int count) {
    int active = 0;
    for (int i = first; i < first+count; ++i)
        if (observation[static_cast<std::size_t>(i)] > 0.001f) ++active;
    return active;
}
void sensorGroup(Canvas& out, int y, const std::string& label, const Observation& observation,
                 int first, int count, Color color) {
    out.text(524,y,label,Muted);
    // Leave a visible gap after the widest label ("SOCIAL / MEM") so its
    // last glyph never touches the meter in the compact inspector.
    out.rect(600,y+1,64,5,Dark);
    out.rect(600,y+1,static_cast<int>(64*sensorMean(observation,first,count)),5,color);
    out.text(671,y,std::to_string(activeSensors(observation,first,count))+
             "/"+std::to_string(count),color);
}
void drawInspector(Canvas& out, const Simulation& sim, const View& view) {
    out.rect(512,56,240,224,Panel);
    out.text(524,66,"INSIDE A MIND",Teal);
    const auto* c = selected(sim,view);
    if (!c) {
        out.text(524,91,"SELECT A CITIZEN",Cream);
        out.wrap(524,111,"CLICK ANYWHERE ON THE MAP TO FOLLOW THE NEAREST LIVING CITIZEN.",35,3);
        out.text(524,168,"INDIVIDUAL POLICIES ON A",Gold);
        out.text(524,182,"SOCIETY CORE 82>2048>2560>2560>12",Gold);
        out.wrap(524,202,std::to_string(InputCount)+" LIVE SENSORS + "+std::to_string(ActionCount)+
                 " ADVISORY CHANNELS. "+std::to_string(HiddenOneCount)+" THEN "+std::to_string(HiddenTwoCount)+
                 " HIDDEN NEURONS. 12 POSSIBLE ACTIONS. LEARNING AFTER EVERY DECISION.",35,4);
        return;
    }
    out.text(524,85,"#"+std::to_string(c->id)+" / CLAN "+std::to_string(c->clan+1),clanColor(c->clan));
    out.text(680,85,"GEN "+std::to_string(c->generation),Muted);
    if (!c->alive) out.text(524,101,"THIS CITIZEN HAS DIED",Red);
    else out.text(524,101,"AGE "+decimal(static_cast<float>(c->age)/TicksPerDay,1)+
                  " D / "+shortened(actionName(c->action),12),Muted);
    compactMeter(out,524,116,"HEALTH",c->health,Teal);
    compactMeter(out,636,116,"HUNGER",c->hunger,Gold);
    compactMeter(out,524,130,"THIRST",c->thirst,0xff91d3e6);
    compactMeter(out,636,130,"ENERGY",c->energy,Teal);
    // Keep the model summary readable at the native 768px layout. The former
    // one-line topology extended beyond the inspector on the right edge.
    out.text(524,145,"LOCAL: "+std::to_string(InputCount)+" SENSORS + "+
             std::to_string(ActionCount)+" ADVICE",Muted);
    out.text(524,158,"SOCIETY CORE 82>2048>2560>2560>12",Gold);
    out.text(524,172,"12.0M PARAMS, PER CIVILIZATION",Muted);

    const Observation observation = sim.observe(*c);
    int valid = 0;
    for (float value : observation)
        if (std::isfinite(value) && value >= 0.0f && value <= 1.0f) ++valid;
    out.text(524,186,"LIVE SENSOR FIELD",Gold);
    // These are summaries only: no inspector interaction can influence a
    // citizen. The groups mirror the stable observation contract in order.
    sensorGroup(out,196,"SELF DRIVE",observation,0,20,Gold);
    sensorGroup(out,205,"SOC / TIME",observation,20,12,Teal);
    sensorGroup(out,214,"LOCAL TILE",observation,32,9,0xff91d3e6);
    sensorGroup(out,223,"NEAR FIELDS",observation,41,22,0xffa9d88b);
    sensorGroup(out,232,"SOCIAL / MEM",observation,63,19,0xffefad7c);
    const Color quality = valid == InputCount ? Teal : Red;
    out.text(524,248,"DATA "+std::to_string(valid)+"/"+std::to_string(InputCount)+
             " VALID / "+std::to_string(activeSensors(observation,0,InputCount))+" LIVE",quality);

    const auto values = c->brain.predict(compose(observation, sim.advice()));
    const auto legal = sim.legalActions(*c);
    std::array<int,ActionCount> ordered{};
    for (int i = 0; i < ActionCount; ++i) ordered[i] = i;
    std::stable_sort(ordered.begin(),ordered.end(),[&](int a, int b) {
        if (legal[a] != legal[b]) return legal[a] > legal[b];
        return values[a] > values[b];
    });
    const int best = ordered[0];
    out.text(524,262,"Q "+shortened(actionName(static_cast<Action>(best)),10),Gold);
    out.rect(604,263,65,5,Dark);
    // Q values are expected discounted rewards, not probabilities.
    const float length = 0.5f+0.5f*std::tanh(values[best]);
    out.rect(604,263,static_cast<int>(65*length),5,Gold);
    out.text(676,262,decimal(values[best]),Gold);
    out.text(524,274,"Q / "+std::to_string(c->brain.updates())+" LEARNING",Muted);
}

void drawEvents(Canvas& out, const Simulation& sim) {
    out.rect(512,292,240,144,Panel);
    out.text(524,303,"THE CHRONICLE",Teal);
    out.text(686,303,"0 - 100",Muted);
    const auto& events = sim.events();
    if (events.empty()) { out.text(524,329,"A WORLD ABOUT TO BEGIN.",Muted); return; }
    for (std::size_t i = 0; i < std::min<std::size_t>(3,events.size()); ++i) {
        const auto& event = events[events.size()-1-i];
        const int y = 324+static_cast<int>(i)*35;
        const Color color = event.score >= 75 ? Red : event.score >= 40 ? Gold : Teal;
        out.rect(524,y,26,14,blend(Panel,color,0.17f));
        out.text(527,y+4,std::to_string(event.score),color);
        out.text(558,y,shortened(event.kind,20),Cream);
        out.wrap(558,y+11,event.text,31,2,Muted);
    }
}

void drawHistory(Canvas& out, const Simulation& sim) {
    out.rect(16,452,736,43,Panel);
    out.text(26,461,"POPULATION",Muted);
    out.text(26,477,std::to_string(sim.stats().population)+" LIVING",Teal);
    constexpr int x = 127, y = 460, width = 457, height = 25;
    out.line(x,y+height,x+width,y+height,Edge);
    const auto& history = sim.history();
    int maximum = 1;
    for (const auto& p : history) maximum = std::max(maximum,p.population);
    maximum = std::max(maximum,sim.stats().population);
    if (!history.empty()) {
        const auto first = history.front().tick;
        const auto end = std::max<std::uint64_t>(first+1,sim.tick());
        int previousX = x;
        int previousY = y+height-history.front().population*height/maximum;
        for (const auto& point : history) {
            const int xx = x+static_cast<int>((point.tick-first)*width/(end-first));
            const int yy = y+height-point.population*height/maximum;
            out.line(previousX,previousY,xx,yy,Teal);
            previousX = xx; previousY = yy;
        }
        out.line(previousX,previousY,x+width,y+height-sim.stats().population*height/maximum,Teal);
    }
    out.text(604,461,"BIRTHS  "+std::to_string(sim.stats().births),Muted);
    out.text(604,477,"DEATHS  "+std::to_string(sim.stats().deaths),Muted);
}

void drawObservation(Canvas& out, const Simulation& sim, const View& view, bool started) {
    out.clear(Ink);
    out.rect(16,15,4,22,Teal); out.rect(23,22,4,15,Gold); out.rect(30,28,4,9,Red);
    out.text(44,16,"PIXEL SOCIETY",Cream,2);
    out.text(44,36,"SET THE CONDITIONS. WATCH LIFE UNFOLD.",Muted);
    out.rect(518,18,4,4,started ? Teal : Gold);
    out.text(531,17,started ? "LIVE / 5 TICKS PER SECOND" : "A WORLD IN WAITING",started ? Teal : Gold);
    out.text(518,33,"DAY "+std::to_string(sim.tick()/TicksPerDay+1)+" / "+sim.seasonName()+" / T "+std::to_string(sim.tick()),Muted);
    metric(out,16,"CITIZENS",std::to_string(sim.stats().population),Teal);
    metric(out,114,"WELLBEING",std::to_string(percent(sim.stats().wellbeing))+"%",Gold);
    metric(out,212,"HOMES",std::to_string(sim.stats().homes));
    metric(out,310,"FARMS",std::to_string(sim.stats().farms));
    metric(out,408,"GENERATION",std::to_string(sim.stats().generation));
    constexpr std::array<const char*,4> labels{"1 LANDSCAPE","2 RESOURCES","3 CLANS","4 BORDERS"};
    for (int i = 0; i < 4; ++i) {
        if (view.layer == i) out.rect(16+i*104,99,98,14,Edge);
        out.text(22+i*104,102,labels[i],view.layer == i ? Cream : Muted);
    }
    out.text(446,103,"TAB: NEXT / H: GUIDE",Muted);
    drawWorld(out,sim,view); drawInspector(out,sim,view); drawEvents(out,sim); drawHistory(out,sim);
    out.text(16,441,view.layer == 1 ? "SOIL: BROWN > TEAL    CITIZENS: TEAL > RED = NEED" : view.layer == 3 ? "TINT: CLAIMED LAND    LINES: CIVILISATION BORDERS" : "CLICK TO FOLLOW / EACH PIXEL CITIZEN HAS ITS OWN LEARNING BRAIN",Muted);
    out.text(16,501,"SEED "+std::to_string(sim.config().seed)+" / AUTONOMOUS AFTER START",Muted);
    out.text(584,501,"H GUIDE / ESC QUIT",Muted);
}

std::string settingValue(const Config& config, int field) {
    switch (field) {
    case 0: return std::to_string(config.seed);
    case 1: return std::to_string(config.founders);
    case 2: return std::to_string(percent(config.fertility))+"%";
    case 3: return std::to_string(percent(config.cooperation))+"%";
    case 5: return nameOf(config.shape);
    case 6: return nameOf(config.worldSize);
    default: return std::to_string(percent(config.hazards))+"%";
    }
}
void adjust(Config& config, int field, int direction) {
    switch (field) {
    case 0: config.seed += static_cast<std::uint32_t>(direction); break;
    case 1: config.founders = std::clamp(config.founders+direction*8,2,PopulationLimit); break;
    case 2: config.fertility = std::clamp(config.fertility+direction*0.05f,0.0f,1.0f); break;
    case 3: config.cooperation = std::clamp(config.cooperation+direction*0.05f,0.0f,1.0f); break;
    case 4: config.hazards = std::clamp(config.hazards+direction*0.05f,0.0f,1.0f); break;
    case 5: config.shape = static_cast<WorldShape>((static_cast<int>(config.shape)+direction+5)%5); break;
    case 6: config.worldSize = static_cast<WorldSize>((static_cast<int>(config.worldSize)+direction+5)%5); break;
    }
}
void darken(Canvas& out) {
    for (auto& pixel : out.pixels) pixel = blend(pixel,Dark,0.72f);
}
void drawSetup(Canvas& out, const Config& config, int selectedField) {
    darken(out);
    // The start control ends at y=435; keep it inside the dialog with a
    // little breathing room below instead of letting it protrude past the frame.
    out.rect(144,95,480,350,Dark); out.frame(144,95,480,350,Edge);
    out.rect(144,95,480,3,Teal);
    out.text(168,113,"THE FIRST CONDITIONS",Cream,2);
    out.text(168,139,"YOUR ONLY INTERVENTION. THEIR ENTIRE FUTURE.",Muted);
    const std::array<const char*,7> labels{"WORLD SEED","FOUNDING CITIZENS","LAND FERTILITY","COOPERATION","NATURAL HAZARDS","WORLD SHAPE","WORLD SIZE"};
    const std::array<const char*,7> hints{"DETERMINISTIC WORLD","INITIAL POPULATION","FOOD REGENERATION","FOUNDERS' SOCIAL TRAIT","FIRE AND RAINSTORMS","ISLAND / ARCHIPELAGO / SEAS / RANGES","TINY TO HUGE MAP"};
    for (int i = 0; i < 7; ++i) {
        const int y = 148+i*30;
        out.rect(164,y,440,27,i == selectedField ? Panel : Dark);
        if (i == selectedField) out.rect(164,y,2,27,Teal);
        out.text(174,y+5,labels[i],i == selectedField ? Cream : Muted);
        out.text(174,y+15,hints[i],Muted);
        out.rect(446,y+4,22,19,Edge); out.text(454,y+10,"-",Cream);
        out.text(475,y+10,settingValue(config,i),i == selectedField ? Gold : Cream);
        out.rect(574,y+4,22,19,Edge); out.text(582,y+10,"+",Cream);
    }
    out.text(168,376,"ARROWS ADJUST / CLICK - + / ENTER TO BEGIN",Muted);
    out.text(168,392,"NO PAUSE. NO ORDERS. JUST OBSERVATION.",Gold);
    out.rect(168,408,432,27,Teal);
    out.text(264,418,"START THE SIMULATION  >",Dark);
}
void drawHelp(Canvas& out) {
    darken(out);
    out.rect(135,93,498,329,Dark); out.frame(135,93,498,329,Edge);
    out.rect(135,93,498,3,Teal);
    out.text(158,113,"AN OBSERVER'S GUIDE",Cream,2);
    out.wrap(158,145,"THE WORLD ADVANCES FIVE TIMES EACH SECOND. EVERY CITIZEN READS 82 LIVE SENSORS PLUS 12 ADVISORY SIGNALS FROM ITS CIVILIZATION'S 12-MILLION-PARAMETER SOCIETY CORE, CHOOSES AN ACTION WITH ITS PERSONAL TWO-HIDDEN-LAYER NETWORK, AND LEARNS FROM THE RESULT.",73,3);
    out.text(158,189,"CLICK MAP",Gold); out.text(284,189,"FOLLOW THE NEAREST LIVING CITIZEN");
    out.text(158,207,"TAB",Gold); out.text(284,207,"FOLLOW THE NEXT LIVING CITIZEN");
    out.text(158,225,"1 / 2 / 3 / 4",Gold); out.text(284,225,"LANDSCAPE / RESOURCES / CLANS / BORDERS");
    out.text(158,243,"H / ESC",Gold); out.text(284,243,"CLOSE GUIDE / ESC AGAIN TO QUIT");
    out.wrap(158,271,"THE INSPECTOR GROUPS SELF, WORLD, LOCAL, NEARBY AND SOCIAL SENSORS. Q VALUES ESTIMATE FUTURE REWARD; THEY ARE NOT PROBABILITIES.",73,3);
    out.wrap(158,316,"EVENT SCORES ALWAYS RUN FROM 0 TO 100: HIGHER MEANS GREATER IMPACT. THE CHRONICLE SHOWS THE LATEST EVENTS. THE RESOURCE LAYER SHOWS SOIL FERTILITY AND CITIZEN NEED. THE BORDERS LAYER TINTS LAND CLAIMED BY EACH CIVILISATION AND LINES ITS FRONTIERS.",73,3);
    out.text(158,365,"THE SIMULATION CONTINUES WHILE THIS GUIDE IS OPEN.",Teal);
    out.text(158,391,"H TO RETURN TO YOUR WORLD",Cream);
}

struct SdlResources {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    ~SdlResources() {
        if (texture) SDL_DestroyTexture(texture);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
    }
};
SDL_Rect viewport(int width, int height) {
    const int scale = std::max(1,std::min(width/Width,height/Height));
    return {(width-Width*scale)/2,(height-Height*scale)/2,Width*scale,Height*scale};
}
void pushKey(SDL_Keycode key) {
    SDL_Event event{}; event.type = SDL_KEYDOWN; event.key.keysym.sym = key; SDL_PushEvent(&event);
}
void pushClick(SDL_Window* window, int x, int y) {
    int w = 0, h = 0; SDL_GetWindowSize(window,&w,&h);
    const auto area = viewport(w,h);
    SDL_Event event{}; event.type = SDL_MOUSEBUTTONDOWN; event.button.button = SDL_BUTTON_LEFT;
    event.button.x = area.x+x*area.w/Width; event.button.y = area.y+y*area.h/Height;
    SDL_PushEvent(&event);
}
bool saveBitmap(Canvas& canvas, const std::string& path) {
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormatFrom(canvas.pixels.data(),Width,Height,32,Width*4,SDL_PIXELFORMAT_ARGB8888);
    if (!surface) return false;
    const bool saved = SDL_SaveBMP(surface,path.c_str()) == 0;
    SDL_FreeSurface(surface);
    return saved;
}
} // namespace

int runUi(const UiOptions& options) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << '\n'; return 1;
    }
    SdlResources sdl;
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY,"0");
    SDL_Rect bounds{0,0,Width,Height}; SDL_GetDisplayUsableBounds(0,&bounds);
    const int initialScale = std::max(1,std::min({2,bounds.w/Width,(bounds.h-60)/Height}));
    sdl.window = SDL_CreateWindow("Pixel Society - an autonomous world",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,
        Width*initialScale,Height*initialScale,SDL_WINDOW_RESIZABLE);
    if (!sdl.window) { std::cerr << "Window creation failed: " << SDL_GetError() << '\n'; return 1; }
    SDL_SetWindowMinimumSize(sdl.window,Width,Height);
    sdl.renderer = SDL_CreateRenderer(sdl.window,-1,SDL_RENDERER_ACCELERATED);
    if (!sdl.renderer) sdl.renderer = SDL_CreateRenderer(sdl.window,-1,SDL_RENDERER_SOFTWARE);
    if (!sdl.renderer) { std::cerr << "Renderer creation failed: " << SDL_GetError() << '\n'; return 1; }
    sdl.texture = SDL_CreateTexture(sdl.renderer,SDL_PIXELFORMAT_ARGB8888,SDL_TEXTUREACCESS_STREAMING,Width,Height);
    if (!sdl.texture) { std::cerr << "Pixel canvas creation failed: " << SDL_GetError() << '\n'; return 1; }

    using Clock = std::chrono::steady_clock;
    Config draft = options.config;
    auto sim = std::make_unique<Simulation>(draft);
    const auto setupDigest = options.smokeTest ? sim->digest() : 0;
    View view;
    Canvas canvas;
    FixedTicker ticker;
    bool running = true, started = false;
    int selectedField = 0, smokePhase = 0;
    bool smokeSelected = false, smokeLayers = false, smokeGuide = false;
    bool smokeSetupFrozen = false;
    Clock::time_point last = Clock::now(), start = last;
    std::chrono::nanoseconds totalElapsed{};
    auto begin = [&] {
        sim = std::make_unique<Simulation>(draft);
        started = true;
        last = start = Clock::now();
        nextCitizen(*sim,view);
    };
    while (running) {
        const auto frameStart = Clock::now();
        if (options.smokeTest && smokePhase == 0) {
            // Exercise all new setup paths before start: founders, shape and
            // size.  The later reference simulation must receive exactly the
            // same frozen draft, proving these controls remain setup-only.
            pushKey(SDLK_DOWN); pushKey(SDLK_RIGHT);
            pushKey(SDLK_DOWN); pushKey(SDLK_DOWN); pushKey(SDLK_DOWN); pushKey(SDLK_DOWN); pushKey(SDLK_RIGHT);
            pushKey(SDLK_DOWN); pushKey(SDLK_RIGHT);
            smokePhase = -1;
        }
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) { running = false; continue; }
            if (event.type == SDL_KEYDOWN && !event.key.repeat) {
                const auto key = event.key.keysym.sym;
                if (key == SDLK_ESCAPE) {
                    if (view.help) view.help = false; else running = false;
                } else if (!started) {
                    if (key == SDLK_UP) selectedField = (selectedField+6)%7;
                    if (key == SDLK_DOWN || key == SDLK_TAB) selectedField = (selectedField+1)%7;
                    if (key == SDLK_LEFT) adjust(draft,selectedField,-1);
                    if (key == SDLK_RIGHT) adjust(draft,selectedField,1);
                    if (key == SDLK_RETURN || key == SDLK_KP_ENTER) begin();
                } else {
                    if (key >= SDLK_1 && key <= SDLK_4) view.layer = static_cast<int>(key-SDLK_1);
                    if (key == SDLK_TAB) nextCitizen(*sim,view);
                    if (key == SDLK_h || key == SDLK_F1) view.help = !view.help;
                }
            }
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                int w = 0, h = 0; SDL_GetWindowSize(sdl.window,&w,&h);
                const auto area = viewport(w,h);
                if (!inside(event.button.x,event.button.y,area.x,area.y,area.w,area.h)) continue;
                const int x = (event.button.x-area.x)*Width/area.w, y = (event.button.y-area.y)*Height/area.h;
                if (!started) {
                    for (int i = 0; i < 7; ++i) {
                        const int yy = 148+i*30;
                        if (inside(x,y,164,yy,440,27)) selectedField = i;
                        if (inside(x,y,446,yy+4,22,19)) adjust(draft,i,-1);
                        if (inside(x,y,574,yy+4,22,19)) adjust(draft,i,1);
                    }
                    if (inside(x,y,168,408,432,27)) begin();
                } else if (!view.help) {
                    const MapView map = mapView(*sim);
                    if (inside(x,y,map.x,map.y,map.w,map.h))
                        selectAt(*sim,view,(x-map.x)/map.cell,(y-map.y)/map.cell);
                    for (int i = 0; i < 4; ++i)
                        if (inside(x,y,16+i*104,99,98,14)) view.layer = i;
                }
            }
        }
        const auto now = Clock::now();
        if (started) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now-last);
            totalElapsed += elapsed;
            ticker.advance(elapsed,[&] { sim->step(); });
        }
        last = now;
        if (options.smokeTest && !started && smokePhase == -1 && now-start >= std::chrono::milliseconds(300)) {
            smokeSetupFrozen = sim->tick() == 0 && sim->digest() == setupDigest;
            pushClick(sdl.window,384,421);
            smokePhase = 1;
        }
        if (options.smokeTest && started) {
            const auto elapsed = now-start;
            if (smokePhase == 1 && elapsed >= std::chrono::milliseconds(300)) {
                const auto* c = selected(*sim,view);
                if (c) {
                    const MapView map = mapView(*sim);
                    pushClick(sdl.window,map.x+c->x*map.cell+2,map.y+c->y*map.cell+2);
                }
                pushKey(SDLK_TAB); pushKey(SDLK_2); smokePhase = 2;
            } else if (smokePhase == 2 && elapsed >= std::chrono::milliseconds(600)) {
                smokeSelected = selected(*sim,view) != nullptr;
                smokeLayers = view.layer == 1;
                pushKey(SDLK_3); pushKey(SDLK_h); smokePhase = 3;
            } else if (smokePhase == 3 && elapsed >= std::chrono::milliseconds(900)) {
                smokeLayers = smokeLayers && view.layer == 2;
                smokeGuide = view.help;
                pushKey(SDLK_4); smokePhase = 4;
            } else if (smokePhase == 4 && elapsed >= std::chrono::milliseconds(1200)) {
                smokeLayers = smokeLayers && view.layer == 3;
                pushKey(SDLK_h); pushKey(SDLK_1);
                // Former setup keys, including ENTER, must have no effect on a live world.
                pushKey(SDLK_DOWN); pushKey(SDLK_RIGHT); pushKey(SDLK_RETURN);
                pushKey(SDLK_UP); pushKey(SDLK_LEFT); pushKey(SDLK_SPACE);
                smokePhase = 5;
            } else if (smokePhase == 5 && elapsed >= std::chrono::milliseconds(2200)) {
                running = false;
            }
        }
        drawObservation(canvas,*sim,view,started);
        if (!started) drawSetup(canvas,draft,selectedField);
        if (view.help) drawHelp(canvas);
        SDL_UpdateTexture(sdl.texture,nullptr,canvas.pixels.data(),Width*static_cast<int>(sizeof(Color)));
        int w = 0, h = 0; SDL_GetRendererOutputSize(sdl.renderer,&w,&h);
        const auto area = viewport(w,h);
        SDL_SetRenderDrawColor(sdl.renderer,7,18,23,255); SDL_RenderClear(sdl.renderer);
        SDL_RenderCopy(sdl.renderer,sdl.texture,nullptr,&area); SDL_RenderPresent(sdl.renderer);
        std::this_thread::sleep_until(frameStart+std::chrono::nanoseconds(1'000'000'000/60));
    }
    if (!options.screenshotPath.empty() && !saveBitmap(canvas,options.screenshotPath)) {
        std::cerr << "Screenshot could not be saved: " << SDL_GetError() << '\n'; return 1;
    }
    if (options.smokeTest) {
        Simulation reference(draft);
        for (std::uint64_t i = 0; i < sim->tick(); ++i) reference.step();
        const auto expected = static_cast<std::uint64_t>(totalElapsed/FixedTicker::interval);
        const bool setupEdited = draft.founders == std::clamp(options.config.founders+8,2,PopulationLimit) &&
            draft.shape != options.config.shape && draft.worldSize != options.config.worldSize;
        const bool autonomous = reference.digest() == sim->digest();
        const auto& actual = sim->config();
        const bool setupLocked = actual.seed == draft.seed && actual.founders == draft.founders &&
            actual.fertility == draft.fertility && actual.cooperation == draft.cooperation && actual.hazards == draft.hazards &&
            actual.shape == draft.shape && actual.worldSize == draft.worldSize;
        const bool success = started && setupEdited && smokeSetupFrozen && setupLocked && smokeSelected && smokeLayers && smokeGuide &&
            !view.help && view.layer == 0 && sim->tick() >= 11 && sim->tick() == expected && autonomous;
        std::cout << "UI smoke: " << (success ? "PASS" : "FAIL") << "; ticks=" << sim->tick()
                  << "; expected_at_5Hz=" << expected << "; setup=" << setupEdited
                  << "; setup_frozen=" << smokeSetupFrozen << "; setup_locked=" << setupLocked
                  << "; selection=" << smokeSelected << "; layers=" << smokeLayers
                  << "; guide=" << smokeGuide << "; observer_preserved_world=" << autonomous << '\n';
        return success ? 0 : 1;
    }
    return 0;
}
} // namespace pixels
